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
def _class_reduce(obj):
    """Select the reducer depending on the dynamic nature of the class obj"""
    if obj is type(None):
        return (type, (None,))
    elif obj is type(Ellipsis):
        return (type, (Ellipsis,))
    elif obj is type(NotImplemented):
        return (type, (NotImplemented,))
    elif obj in _BUILTIN_TYPE_NAMES:
        return (_builtin_type, (_BUILTIN_TYPE_NAMES[obj],))
    elif not _should_pickle_by_reference(obj):
        return _dynamic_class_reduce(obj)
    return NotImplemented