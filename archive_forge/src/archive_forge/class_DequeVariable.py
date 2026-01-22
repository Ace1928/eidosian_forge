import collections
import functools
import inspect
import operator
import types
from typing import Dict, List, Optional
import torch
import torch.fx
from ..._guards import Source
from .. import polyfill, variables
from ..bytecode_transformation import create_call_function, create_instruction
from ..exc import unimplemented
from ..source import AttrSource, GetItemSource
from ..utils import (
from .base import MutableLocal, VariableTracker
from .constant import ConstantVariable
from .functions import UserFunctionVariable, UserMethodVariable
class DequeVariable(CommonListMethodsVariable):

    def python_type(self):
        return collections.deque

    def reconstruct(self, codegen):
        assert 'deque' not in codegen.tx.f_globals
        codegen.append_output(codegen.create_load_python_module(collections.deque, True))
        codegen.foreach(self.items)
        return create_call_function(len(self.items), False)

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        if name == '__setitem__' and self.mutable_local and args and args[0].is_python_constant():
            assert not kwargs
            key, value = args
            assert key.is_python_constant() and isinstance(key.as_python_constant(), int)
            items = list(self.items)
            items[key.as_python_constant()] = value
            result = DequeVariable(items)
            return tx.replace_all(self, result)
        elif name == 'extendleft' and self.mutable_local:
            assert not kwargs
            arg, = args
            return tx.replace_all(self, DequeVariable(list(arg.unpack_var_sequence(tx)) + list(self.items)))
        elif name == 'popleft' and self.mutable_local:
            assert not args
            assert not kwargs
            items = collections.deque(self.items)
            result = items.popleft()
            tx.replace_all(self, DequeVariable(list(items)))
            return result
        elif name == 'appendleft' and self.mutable_local:
            assert not kwargs
            return tx.replace_all(self, DequeVariable([args[0]] + list(self.items)))
        else:
            return super().call_method(tx, name, args, kwargs)