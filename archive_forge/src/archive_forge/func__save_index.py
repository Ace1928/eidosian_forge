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
def _save_index(self, overloads):
    data = (self._source_stamp, overloads)
    data = self._dump(data)
    with self._open_for_write(self._index_path) as f:
        pickle.dump(self._version, f, protocol=-1)
        f.write(data)
    _cache_log('[cache] index saved to %r', self._index_path)