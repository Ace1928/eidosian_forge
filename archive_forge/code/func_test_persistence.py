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
def test_persistence(tmpdir):
    memory = Memory(location=tmpdir.strpath, verbose=0)
    g = memory.cache(f)
    output = g(1)
    h = pickle.loads(pickle.dumps(g))
    func_id, args_id = h._get_output_identifiers(1)
    output_dir = os.path.join(h.store_backend.location, func_id, args_id)
    assert os.path.exists(output_dir)
    assert output == h.store_backend.load_item([func_id, args_id])
    memory2 = pickle.loads(pickle.dumps(memory))
    assert memory.store_backend.location == memory2.store_backend.location
    memory = Memory(location=None, verbose=0)
    pickle.loads(pickle.dumps(memory))
    g = memory.cache(f)
    gp = pickle.loads(pickle.dumps(g))
    gp(1)