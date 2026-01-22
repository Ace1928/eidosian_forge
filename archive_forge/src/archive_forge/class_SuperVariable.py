import collections
import dataclasses
import functools
import inspect
import itertools
import operator
import sys
import types
from typing import Dict, List
import torch._C
import torch._numpy as tnp
from .. import config, polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource, GetItemSource, ODictGetItemSource, TypeSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .dicts import DefaultDictVariable
from .functions import (
from .user_defined import UserDefinedObjectVariable
class SuperVariable(VariableTracker):

    def __init__(self, typevar, objvar=None, specialized=False, **kwargs):
        super().__init__(**kwargs)
        self.typevar = typevar
        self.objvar = objvar
        self.specialized = specialized

    def reconstruct(self, codegen):
        codegen(variables.BuiltinVariable(super))
        codegen(self.typevar)
        if self.objvar is not None:
            codegen(self.objvar)
            return create_call_function(2, True)
        else:
            return create_call_function(1, True)

    def _resolved_getattr_and_source(self, tx, name):
        assert self.objvar, '1-arg super not implemented'
        if self.specialized:
            return getattr(self.typevar.as_python_constant(), name)
        search_type = self.typevar.as_python_constant()
        type_to_use = self.objvar.python_type()
        type_to_use_source = TypeSource(self.objvar.source) if self.objvar.source else None
        if issubclass(type_to_use, type):
            type_to_use = self.objvar.value
            type_to_use_source = self.objvar.source
        source = None
        if self.objvar.source is not None:
            search_mro = type_to_use.__mro__
            start_index = search_mro.index(search_type) + 1
            for index in range(start_index, len(search_mro)):
                if hasattr(search_mro[index], name):
                    source = AttrSource(GetItemSource(AttrSource(type_to_use_source, '__mro__'), index), name)
                    break
        return (getattr(super(search_type, type_to_use), name), source)

    def var_getattr(self, tx, name: str) -> 'VariableTracker':
        value, source = self._resolved_getattr_and_source(self, name)
        if not variables.ConstantVariable.is_literal(value):
            return GetAttrVariable(self, name)
        if source:
            install_guard(source.make_guard(GuardBuilder.CONSTANT_MATCH))
            return variables.ConstantVariable.create(value, source=source)
        return variables.ConstantVariable.create(value)

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        inner_fn, source = self._resolved_getattr_and_source(self, name)
        if inner_fn is object.__init__:
            return LambdaVariable(identity)
        elif inner_fn is torch.nn.Module.__init__:
            objvar = self.objvar
            from ..side_effects import AttributeMutationNew
            if isinstance(objvar, variables.UserDefinedObjectVariable) and isinstance(objvar.mutable_local, AttributeMutationNew) and (not (args or kwargs)):
                tx.output.side_effects.store_attr(objvar, '__call_nn_module_init', variables.ConstantVariable.create(True))
                return variables.ConstantVariable.create(None)
            else:
                unimplemented('super() nn.Module.__init__')
        elif isinstance(inner_fn, types.FunctionType):
            return variables.UserFunctionVariable(inner_fn, source=source).call_function(tx, [self.objvar] + args, kwargs)
        elif isinstance(inner_fn, types.MethodType):
            return variables.UserMethodVariable(inner_fn.__func__, self.objvar, source=source).call_function(tx, args, kwargs)
        elif inner_fn is collections.OrderedDict.__getitem__ and isinstance(self.objvar, variables.UserDefinedObjectVariable) and self.objvar.source and (len(args) == 1) and (len(kwargs) == 0) and args[0].is_python_constant():
            from .builder import VariableBuilder
            key = args[0].as_python_constant()
            return VariableBuilder(tx, ODictGetItemSource(self.objvar.source, key))(collections.OrderedDict.__getitem__(self.objvar.value, key))
        elif inner_fn in (collections.OrderedDict.__setitem__, object.__setattr__) and isinstance(self.objvar, variables.CustomizedDictVariable) and args and variables.ConstDictVariable.is_valid_key(args[0]) and self.objvar.mutable_local:
            assert not kwargs and len(args) == 2
            k = variables.ConstDictVariable.get_key(args[0])
            newval = dict(self.objvar.items)
            newval[k] = args[1]
            return tx.replace_all(self.objvar, self.objvar.modifed(newval))
        else:
            unimplemented(f'non-function or method super: {inner_fn}')