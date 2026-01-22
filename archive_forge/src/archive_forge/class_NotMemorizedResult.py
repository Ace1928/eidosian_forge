from __future__ import with_statement
import logging
import os
from textwrap import dedent
import time
import pathlib
import pydoc
import re
import functools
import traceback
import warnings
import inspect
import weakref
from datetime import timedelta
from tokenize import open as open_py_source
from . import hashing
from .func_inspect import get_func_code, get_func_name, filter_args
from .func_inspect import format_call
from .func_inspect import format_signature
from .logger import Logger, format_time, pformat
from ._store_backends import StoreBackendBase, FileSystemStoreBackend
from ._store_backends import CacheWarning  # noqa
class NotMemorizedResult(object):
    """Class representing an arbitrary value.

    This class is a replacement for MemorizedResult when there is no cache.
    """
    __slots__ = ('value', 'valid')

    def __init__(self, value):
        self.value = value
        self.valid = True

    def get(self):
        if self.valid:
            return self.value
        else:
            raise KeyError('No value stored.')

    def clear(self):
        self.valid = False
        self.value = None

    def __repr__(self):
        if self.valid:
            return '{class_name}({value})'.format(class_name=self.__class__.__name__, value=pformat(self.value))
        else:
            return self.__class__.__name__ + ' with no value'

    def __getstate__(self):
        return {'valid': self.valid, 'value': self.value}

    def __setstate__(self, state):
        self.valid = state['valid']
        self.value = state['value']