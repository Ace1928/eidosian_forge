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
def _class_getstate(obj):
    clsdict = _extract_class_dict(obj)
    clsdict.pop('__weakref__', None)
    if issubclass(type(obj), abc.ABCMeta):
        clsdict.pop('_abc_cache', None)
        clsdict.pop('_abc_negative_cache', None)
        clsdict.pop('_abc_negative_cache_version', None)
        registry = clsdict.pop('_abc_registry', None)
        if registry is None:
            if hasattr(abc, '_get_dump'):
                clsdict.pop('_abc_impl', None)
                registry, _, _, _ = abc._get_dump(obj)
                clsdict['_abc_impl'] = [subclass_weakref() for subclass_weakref in registry]
            else:
                clsdict['_abc_impl'] = None
        else:
            clsdict['_abc_impl'] = [type_ for type_ in registry]
    if '__slots__' in clsdict:
        if isinstance(obj.__slots__, str):
            clsdict.pop(obj.__slots__)
        else:
            for k in obj.__slots__:
                clsdict.pop(k, None)
    clsdict.pop('__dict__', None)
    return (clsdict, {})