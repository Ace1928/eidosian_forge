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
def _save_with_postproc(pickler, reduction, is_pickler_dill=None, obj=Getattr.NO_DEFAULT, postproc_list=None):
    if obj is Getattr.NO_DEFAULT:
        obj = Reduce(reduction)
    if is_pickler_dill is None:
        is_pickler_dill = is_dill(pickler, child=True)
    if is_pickler_dill:
        if postproc_list is None:
            postproc_list = []
        if id(obj) in pickler._postproc:
            name = '%s.%s ' % (obj.__module__, getattr(obj, '__qualname__', obj.__name__)) if hasattr(obj, '__module__') else ''
            warnings.warn('Cannot pickle %r: %shas recursive self-references that trigger a RecursionError.' % (obj, name), PicklingWarning)
            pickler.save_global(obj)
            return
        pickler._postproc[id(obj)] = postproc_list
    pickler.save_reduce(*reduction, obj=obj)
    if is_pickler_dill:
        postproc = pickler._postproc.pop(id(obj))
        for reduction in reversed(postproc):
            if reduction[0] is _setitems:
                dest, source = reduction[1]
                if source:
                    pickler.write(pickler.get(pickler.memo[id(dest)][0]))
                    pickler._batch_setitems(iter(source.items()))
                else:
                    continue
            else:
                pickler.save_reduce(*reduction)
            pickler.write(POP)