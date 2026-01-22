from abc import ABCMeta, abstractmethod, abstractproperty
import contextlib
import errno
import hashlib
import inspect
import itertools
import os
import pickle
import sys
import tempfile
import uuid
import warnings
from numba.misc.appdirs import AppDirs
import numba
from numba.core.errors import NumbaWarning
from numba.core.base import BaseContext
from numba.core.codegen import CodeLibrary
from numba.core.compiler import CompileResult
from numba.core import config, compiler
from numba.core.serialize import dumps
def _save_overload(self, sig, data):
    if not self._enabled:
        return
    if not self._impl.check_cachable(data):
        return
    self._impl.locator.ensure_cache_path()
    key = self._index_key(sig, data.codegen)
    data = self._impl.reduce(data)
    self._cache_file.save(key, data)