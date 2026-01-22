from __future__ import annotations
import warnings
from .logger import adapter as logger
from .logger import trace as _trace
import os
import sys
import builtins as __builtin__
from pickle import _Pickler as StockPickler, Unpickler as StockUnpickler
from pickle import GLOBAL, POP
from _thread import LockType
from _thread import RLock as RLockType
from types import CodeType, FunctionType, MethodType, GeneratorType, \
from types import MappingProxyType as DictProxyType, new_class
from pickle import DEFAULT_PROTOCOL, HIGHEST_PROTOCOL, PickleError, PicklingError, UnpicklingError
import __main__ as _main_module
import marshal
import gc
import abc
import dataclasses
from weakref import ReferenceType, ProxyType, CallableProxyType
from collections import OrderedDict
from enum import Enum, EnumMeta
from functools import partial
from operator import itemgetter, attrgetter
import importlib.machinery
from types import GetSetDescriptorType, ClassMethodDescriptorType, \
from io import BytesIO as StringIO
from socket import socket as SocketType
from multiprocessing.reduction import _reduce_socket as reduce_socket
import inspect
import typing
from . import _shims
from ._shims import Reduce, Getattr
def _proxy_helper(obj):
    """get memory address of proxy's reference object"""
    _repr = repr(obj)
    try:
        _str = str(obj)
    except ReferenceError:
        return id(None)
    if _str == _repr:
        return id(obj)
    try:
        address = int(_str.rstrip('>').split(' at ')[-1], base=16)
    except ValueError:
        if not IS_PYPY:
            address = int(_repr.rstrip('>').split(' at ')[-1], base=16)
        else:
            objects = iter(gc.get_objects())
            for _obj in objects:
                if repr(_obj) == _str:
                    return id(_obj)
            msg = "Cannot reference object for proxy at '%s'" % id(obj)
            raise ReferenceError(msg)
    return address