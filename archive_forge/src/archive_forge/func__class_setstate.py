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
def _class_setstate(obj, state):
    state, slotstate = state
    registry = None
    for attrname, attr in state.items():
        if attrname == '_abc_impl':
            registry = attr
        else:
            setattr(obj, attrname, attr)
    if registry is not None:
        for subclass in registry:
            obj.register(subclass)
    return obj