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
def _create_filehandle(name, mode, position, closed, open, strictio, fmode, fdata):
    names = {'<stdin>': sys.__stdin__, '<stdout>': sys.__stdout__, '<stderr>': sys.__stderr__}
    if name in list(names.keys()):
        f = names[name]
    elif name == '<tmpfile>':
        f = os.tmpfile()
    elif name == '<fdopen>':
        import tempfile
        f = tempfile.TemporaryFile(mode)
    else:
        try:
            exists = os.path.exists(name)
        except Exception:
            exists = False
        if not exists:
            if strictio:
                raise FileNotFoundError("[Errno 2] No such file or directory: '%s'" % name)
            elif 'r' in mode and fmode != FILE_FMODE:
                name = '<fdopen>'
            current_size = 0
        else:
            current_size = os.path.getsize(name)
        if position > current_size:
            if strictio:
                raise ValueError('invalid buffer size')
            elif fmode == CONTENTS_FMODE:
                position = current_size
        try:
            if fmode == FILE_FMODE:
                f = open(name, mode if 'w' in mode else 'w')
                f.write(fdata)
                if 'w' not in mode:
                    f.close()
                    f = open(name, mode)
            elif name == '<fdopen>':
                import tempfile
                f = tempfile.TemporaryFile(mode)
            elif fmode == CONTENTS_FMODE and ('w' in mode or 'x' in mode):
                flags = os.O_CREAT
                if '+' in mode:
                    flags |= os.O_RDWR
                else:
                    flags |= os.O_WRONLY
                f = os.fdopen(os.open(name, flags), mode)
                r = getattr(f, 'buffer', f)
                r = getattr(r, 'raw', r)
                r.name = name
                assert f.name == name
            else:
                f = open(name, mode)
        except (IOError, FileNotFoundError):
            err = sys.exc_info()[1]
            raise UnpicklingError(err)
    if closed:
        f.close()
    elif position >= 0 and fmode != HANDLE_FMODE:
        f.seek(position)
    return f