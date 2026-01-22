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
@contextlib.contextmanager
def _guard_against_spurious_io_errors(self):
    if os.name == 'nt':
        try:
            yield
        except OSError as e:
            if e.errno != errno.EACCES:
                raise
    else:
        yield