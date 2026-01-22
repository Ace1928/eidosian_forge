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
def _open_for_write(self, filepath):
    """
        Open *filepath* for writing in a race condition-free way (hopefully).
        uuid4 is used to try and avoid name collisions on a shared filesystem.
        """
    uid = uuid.uuid4().hex[:16]
    tmpname = '%s.tmp.%s' % (filepath, uid)
    try:
        with open(tmpname, 'wb') as f:
            yield f
        os.replace(tmpname, filepath)
    except Exception:
        try:
            os.unlink(tmpname)
        except OSError:
            pass
        raise