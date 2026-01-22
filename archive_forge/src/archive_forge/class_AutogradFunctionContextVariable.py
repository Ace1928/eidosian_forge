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
class AutogradFunctionContextVariable(UserDefinedObjectVariable):
    """
    Tracks an autograd.Function() context using mutation tracking in side_effects.py
    """
    _nonvar_fields = {'proxy', 'inference', *UserDefinedObjectVariable._nonvar_fields}

    def __init__(self, value, value_type=None, inference=False, proxy=None, saved_tensors=None, **kwargs):
        super().__init__(value=value, value_type=value_type, **kwargs)
        self.inference = inference
        self.proxy = proxy
        self.saved_tensors = saved_tensors

    @staticmethod
    def create(tx):
        proxy = tx.output.create_proxy('call_function', torch.autograd.function.FunctionCtx, tuple(), {})
        out = tx.output.side_effects.track_object_new(None, torch.autograd.function.FunctionCtx, functools.partial(AutogradFunctionContextVariable, inference=True, proxy=proxy, saved_tensors=SavedTensorBox()), {})
        proxy.node.meta['example_value'] = out.value
        return out

    def as_proxy(self):
        if self.proxy is None:
            unimplemented('proxy not set')
        return self.proxy

    def call_method(self, tx, name, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if name != 'save_for_backward':
            unimplemented(f'autograd.Function context method: {name}')
        if self.saved_tensors is None:
            unimplemented('save_for_backward only supported on a newly constructed FunctionCtx')
        if not self.inference:
            assert self.source and (not kwargs)
            tx.output.side_effects.track_save_for_backward(self, args)
        for arg in args:
            self.saved_tensors.tensors.append(arg)
        return variables.ConstantVariable.create(None)

    def var_getattr(self, tx, name):
        if name == 'save_for_backward':
            return LambdaVariable(lambda *args, **kwargs: self.call_method(tx, name, args, kwargs))
        if name == 'saved_tensors' and self.saved_tensors is not None:
            return variables.TupleVariable(list(self.saved_tensors.tensors))
        return super().var_getattr(tx, name)