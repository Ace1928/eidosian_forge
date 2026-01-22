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
def _index_key(self, sig, codegen):
    """
        Compute index key for the given signature and codegen.
        It includes a description of the OS, target architecture and hashes of
        the bytecode for the function and, if the function has a __closure__,
        a hash of the cell_contents.
        """
    codebytes = self._py_func.__code__.co_code
    if self._py_func.__closure__ is not None:
        cvars = tuple([x.cell_contents for x in self._py_func.__closure__])
        cvarbytes = dumps(cvars)
    else:
        cvarbytes = b''
    hasher = lambda x: hashlib.sha256(x).hexdigest()
    return (sig, codegen.magic_tuple(), (hasher(codebytes), hasher(cvarbytes)))