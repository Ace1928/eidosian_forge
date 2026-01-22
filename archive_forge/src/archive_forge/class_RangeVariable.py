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
class RangeVariable(BaseListVariable):

    def __init__(self, items, **kwargs):
        items_to_map = items
        start = variables.ConstantVariable.create(0)
        stop = None
        step = variables.ConstantVariable.create(1)
        if len(items_to_map) == 1:
            stop, = items_to_map
        elif len(items_to_map) == 2:
            start, stop = items_to_map
        elif len(items_to_map) == 3:
            start, stop, step = items_to_map
        else:
            raise AssertionError()
        assert stop is not None
        super().__init__([start, stop, step], **kwargs)

    def python_type(self):
        return range

    def as_python_constant(self):
        return range(*[x.as_python_constant() for x in self.items])

    def as_proxy(self):
        return self.python_type()(*self._as_proxy())

    def unpack_var_sequence(self, tx):
        return [variables.ConstantVariable.create(x) for x in self.as_python_constant()]

    def reconstruct(self, codegen):
        assert 'range' not in codegen.tx.f_globals
        codegen.append_output(codegen.create_load_python_module(range, True))
        codegen.foreach(self.items)
        return create_call_function(3, False)

    def var_getattr(self, tx, name):
        fields = ['start', 'stop', 'step']
        if name not in fields:
            unimplemented(f'range.{name}')
        return self.items[fields.index(name)]