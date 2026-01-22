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