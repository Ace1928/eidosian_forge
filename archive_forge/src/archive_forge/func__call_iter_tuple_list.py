import contextlib
import functools
import inspect
import itertools
import logging
import math
import operator
import types
from collections import defaultdict, OrderedDict
from typing import Dict, List
import torch
from torch import sym_float, sym_int
from .. import config, polyfill, variables
from ..exc import (
from ..guards import GuardBuilder, install_guard
from ..replay_record import DummyModule
from ..source import AttrSource, GetItemSource, is_constant_source, TypeSource
from ..utils import (
from .base import MutableLocal, typestr, VariableTracker
from .constant import ConstantVariable
from .ctx_manager import EventVariable, StreamVariable
from .dicts import ConstDictVariable, DefaultDictVariable, SetVariable
from .lists import (
from .tensor import FakeItemVariable, SymNodeVariable, UnspecializedPythonVariable
from .user_defined import UserDefinedVariable
def _call_iter_tuple_list(self, tx, obj=None, *args, **kwargs):
    if self._dynamic_args(*args, **kwargs):
        return self._dyn_proxy(tx, *args, **kwargs)
    if isinstance(obj, variables.IteratorVariable):
        return obj
    if self.fn == set:
        cls = SetVariable
    else:
        cls = variables.BaseListVariable.cls_for(self.fn)
    if obj is None:
        if cls is SetVariable:
            return cls([], mutable_local=MutableLocal())
        else:
            return cls([], mutable_local=MutableLocal())
    elif obj.has_unpack_var_sequence(tx):
        if obj.source and (not is_constant_source(obj.source)):
            if isinstance(obj, TupleIteratorVariable):
                install_guard(obj.source.make_guard(GuardBuilder.TUPLE_ITERATOR_LEN))
            else:
                install_guard(obj.source.make_guard(GuardBuilder.LIST_LENGTH))
        if cls is SetVariable:
            return cls(list(obj.unpack_var_sequence(tx)), mutable_local=MutableLocal())
        return cls(list(obj.unpack_var_sequence(tx)), mutable_local=MutableLocal())