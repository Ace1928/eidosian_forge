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
class TupleVariable(BaseListVariable):

    def python_type(self):
        return tuple

    def reconstruct(self, codegen):
        codegen.foreach(self.items)
        return [create_instruction('BUILD_TUPLE', arg=len(self.items))]

    def call_method(self, tx, name, args: List['VariableTracker'], kwargs: Dict[str, 'VariableTracker']) -> 'VariableTracker':
        return super().call_method(tx, name, args, kwargs)