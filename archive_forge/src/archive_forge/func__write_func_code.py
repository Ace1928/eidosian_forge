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
def _write_func_code(self, func_code, first_line):
    """ Write the function code and the filename to a file.
        """
    func_id = _build_func_identifier(self.func)
    func_code = u'%s %i\n%s' % (FIRST_LINE_TEXT, first_line, func_code)
    self.store_backend.store_cached_func_code([func_id], func_code)
    is_named_callable = False
    is_named_callable = hasattr(self.func, '__name__') and self.func.__name__ != '<lambda>'
    if is_named_callable:
        func_hash = self._hash_func()
        try:
            _FUNCTION_HASHES[self.func] = func_hash
        except TypeError:
            pass