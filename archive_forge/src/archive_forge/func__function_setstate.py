import _collections_abc
import abc
import copyreg
import io
import itertools
import logging
import sys
import struct
import types
import weakref
import typing
from enum import Enum
from collections import ChainMap, OrderedDict
from .compat import pickle, Pickler
from .cloudpickle import (
def _function_setstate(obj, state):
    """Update the state of a dynamic function.

    As __closure__ and __globals__ are readonly attributes of a function, we
    cannot rely on the native setstate routine of pickle.load_build, that calls
    setattr on items of the slotstate. Instead, we have to modify them inplace.
    """
    state, slotstate = state
    obj.__dict__.update(state)
    obj_globals = slotstate.pop('__globals__')
    obj_closure = slotstate.pop('__closure__')
    slotstate.pop('_cloudpickle_submodules')
    obj.__globals__.update(obj_globals)
    obj.__globals__['__builtins__'] = __builtins__
    if obj_closure is not None:
        for i, cell in enumerate(obj_closure):
            try:
                value = cell.cell_contents
            except ValueError:
                continue
            cell_set(obj.__closure__[i], value)
    for k, v in slotstate.items():
        setattr(obj, k, v)