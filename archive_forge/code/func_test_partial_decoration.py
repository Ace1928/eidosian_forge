import functools
import gc
import logging
import shutil
import os
import os.path
import pathlib
import pickle
import sys
import time
import datetime
import textwrap
import pytest
from joblib.memory import Memory
from joblib.memory import expires_after
from joblib.memory import MemorizedFunc, NotMemorizedFunc
from joblib.memory import MemorizedResult, NotMemorizedResult
from joblib.memory import _FUNCTION_HASHES
from joblib.memory import register_store_backend, _STORE_BACKENDS
from joblib.memory import _build_func_identifier, _store_backend_factory
from joblib.memory import JobLibCollisionWarning
from joblib.parallel import Parallel, delayed
from joblib._store_backends import StoreBackendBase, FileSystemStoreBackend
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.testing import parametrize, raises, warns
from joblib.hashing import hash
@parametrize('ignore, verbose, mmap_mode', [(['x'], 100, 'r'), ([], 10, None)])
def test_partial_decoration(tmpdir, ignore, verbose, mmap_mode):
    """Check cache may be called with kwargs before decorating"""
    memory = Memory(location=tmpdir.strpath, verbose=0)

    @memory.cache(ignore=ignore, verbose=verbose, mmap_mode=mmap_mode)
    def z(x):
        pass
    assert z.ignore == ignore
    assert z._verbose == verbose
    assert z.mmap_mode == mmap_mode