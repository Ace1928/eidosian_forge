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
def _class_getnewargs(obj):
    type_kwargs = {}
    if '__slots__' in obj.__dict__:
        type_kwargs['__slots__'] = obj.__slots__
    __dict__ = obj.__dict__.get('__dict__', None)
    if isinstance(__dict__, property):
        type_kwargs['__dict__'] = __dict__
    return (type(obj), obj.__name__, _get_bases(obj), type_kwargs, _get_or_create_tracker_id(obj), None)