import collections
import contextlib
import functools
import importlib
import inspect
import itertools
import random
import sys
import threading
import types
from typing import Dict, List
import torch._dynamo.config
import torch.nn
from torch._guards import TracingContext
from .. import variables
from ..allowed_functions import is_allowed
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, RandomValueSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .ctx_manager import GenericContextWrappingVariable, NullContextVariable
from .dicts import ConstDictVariable
class UserDefinedClassVariable(UserDefinedVariable):

    def __init__(self, value, **kwargs):
        super().__init__(**kwargs)
        self.value = value

    def as_python_constant(self):
        return self.value

    def python_type(self):
        return type(self.value)

    def var_getattr(self, tx, name: str) -> 'VariableTracker':
        from . import ConstantVariable
        from .builder import VariableBuilder
        source = AttrSource(self.source, name) if self.source is not None else None
        try:
            obj = inspect.getattr_static(self.value, name)
        except AttributeError:
            obj = None
        if isinstance(obj, staticmethod):
            return variables.UserFunctionVariable(obj.__get__(self.value), source=source)
        elif isinstance(obj, classmethod):
            return variables.UserMethodVariable(obj.__func__, self, source=source)
        elif source and inspect.ismemberdescriptor(obj):
            return VariableBuilder(tx, source)(obj.__get__(self.value))
        if name in getattr(self.value, '__dict__', {}) or ConstantVariable.is_literal(obj):
            if source:
                return VariableBuilder(tx, source)(obj)
            elif ConstantVariable.is_literal(obj):
                return ConstantVariable.create(obj)
        return super().var_getattr(tx, name)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if name == '__subclasses__' and len(args) == 0 and (not kwargs) and ('__subclasses__' not in self.value.__dict__):
            options = {'mutable_local': MutableLocal()}
            subs_as_vars: List[VariableTracker] = list()
            for sub in self.value.__subclasses__():
                source = AttrSource(tx.import_source(sub.__module__), sub.__name__)
                subs_as_vars.append(variables.UserDefinedClassVariable(sub, source=source))
            return variables.ListVariable(subs_as_vars, **options)
        return super().call_method(tx, name, args, kwargs)

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        from ..side_effects import SideEffects
        from .builder import SourcelessBuilder
        if self.value is contextlib.nullcontext:
            return NullContextVariable()
        elif issubclass(type(self.value), type) and hasattr(self.value, '__enter__') and hasattr(self.value, '__exit__') and check_constant_args(args, kwargs) and (len(kwargs) == 0):
            unwrapped_args = [x.as_python_constant() for x in args]
            return GenericContextWrappingVariable(unwrapped_args, cm_obj=self.value(*unwrapped_args))
        elif is_namedtuple_cls(self.value):
            fields = namedtuple_fields(self.value)
            field_defaults = self.value._field_defaults
            items = list(args)
            items.extend([None] * (len(fields) - len(items)))
            var_tracker_kwargs = {}
            for field_name, var_tracker in zip(fields, items):
                if var_tracker is None:
                    if field_name in kwargs:
                        field_var = kwargs[field_name]
                    else:
                        assert field_name in field_defaults
                        field_var = SourcelessBuilder()(tx, field_defaults[field_name])
                    var_tracker_kwargs[field_name] = field_var
            for name, value in var_tracker_kwargs.items():
                assert name in fields
                items[fields.index(name)] = value
            assert all((x is not None for x in items))
            return variables.NamedTupleVariable(items, self.value)
        elif inspect.getattr_static(self.value, '__new__', None) in (object.__new__,) and SideEffects.cls_supports_mutation_side_effects(self.value) and self.source:
            var = tx.output.side_effects.track_object_new(self.source, self.value, variables.UnspecializedNNModuleVariable if issubclass(self.value, torch.nn.Module) else UserDefinedObjectVariable, {})
            if inspect.getattr_static(self.value, '__init__', None) is torch.nn.Module.__init__:
                tx.output.side_effects.store_attr(var, '__call_nn_module_init', variables.ConstantVariable.create(True))
                return var
            else:
                var.call_method(tx, '__init__', args, kwargs)
                return var
        elif variables.CustomizedDictVariable.is_matching_cls(self.value):
            options = {'mutable_local': MutableLocal()}
            return variables.CustomizedDictVariable.create(self.value, args, kwargs, options)
        elif variables.DataClassVariable.is_matching_cls(self.value):
            options = {'mutable_local': MutableLocal()}
            return variables.DataClassVariable.create(self.value, args, kwargs, options)
        elif variables.RestrictedListSubclassVariable.is_matching_cls(self.value) and self.source:
            return variables.RestrictedListSubclassVariable(variables.BuiltinVariable(list).call_function(tx, args, kwargs).items, user_cls=self.value, user_cls_source=self.source, mutable_local=MutableLocal())
        return super().call_function(tx, args, kwargs)

    def const_getattr(self, tx, name):
        if name == '__name__':
            return self.value.__name__
        return super().const_getattr(tx, name)