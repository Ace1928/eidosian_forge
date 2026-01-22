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
class ListVariable(CommonListMethodsVariable):

    def python_type(self):
        return list

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction('BUILD_LIST', arg=len(self.items))]

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        if name == '__setitem__' and self.mutable_local and args and args[0].is_python_constant():
            assert not kwargs
            key, value = args
            items = list(self.items)
            if isinstance(key, SliceVariable):
                if not value.has_unpack_var_sequence(tx):
                    unimplemented(f'Missing dynamo support for expanding {value} into a list for slice assignment.')
                items[key.as_python_constant()] = value.unpack_var_sequence(tx)
            else:
                items[key.as_python_constant()] = value
            result = ListVariable(items)
            return tx.replace_all(self, result)
        else:
            return super().call_method(tx, name, args, kwargs)

    def call_hasattr(self, tx, name: str) -> 'VariableTracker':
        if self.python_type() is not list:
            return super().call_hasattr(tx, name)
        return variables.ConstantVariable.create(hasattr([], name))