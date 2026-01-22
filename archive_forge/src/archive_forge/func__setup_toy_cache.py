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
def _setup_toy_cache(tmpdir, num_inputs=10):
    memory = Memory(location=tmpdir.strpath, verbose=0)

    @memory.cache()
    def get_1000_bytes(arg):
        return 'a' * 1000
    inputs = list(range(num_inputs))
    for arg in inputs:
        get_1000_bytes(arg)
    func_id = _build_func_identifier(get_1000_bytes)
    hash_dirnames = [get_1000_bytes._get_output_identifiers(arg)[1] for arg in inputs]
    full_hashdirs = [os.path.join(get_1000_bytes.store_backend.location, func_id, dirname) for dirname in hash_dirnames]
    return (memory, full_hashdirs, get_1000_bytes)