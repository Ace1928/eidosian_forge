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
@parametrize('call_before_reducing', [True, False])
def test_parallel_call_cached_function_defined_in_jupyter(tmpdir, call_before_reducing):
    for session_no in [0, 1]:
        ipython_cell_source = '\n        def f(x):\n            return x\n        '
        ipython_cell_id = '<ipython-input-{}-000000000000>'.format(session_no)
        exec(compile(textwrap.dedent(ipython_cell_source), filename=ipython_cell_id, mode='exec'))
        aliased_f = locals()['f']
        aliased_f.__module__ = '__main__'
        assert aliased_f(1) == 1
        assert aliased_f.__code__.co_filename == ipython_cell_id
        memory = Memory(location=tmpdir.strpath, verbose=0)
        cached_f = memory.cache(aliased_f)
        assert len(os.listdir(tmpdir / 'joblib')) == 1
        f_cache_relative_directory = os.listdir(tmpdir / 'joblib')[0]
        assert 'ipython-input' in f_cache_relative_directory
        f_cache_directory = tmpdir / 'joblib' / f_cache_relative_directory
        if session_no == 0:
            assert os.listdir(f_cache_directory) == ['f']
            assert os.listdir(f_cache_directory / 'f') == []
            if call_before_reducing:
                cached_f(3)
                assert len(os.listdir(f_cache_directory / 'f')) == 2
                res = Parallel(n_jobs=2)((delayed(cached_f)(i) for i in [1, 2]))
            else:
                res = Parallel(n_jobs=2)((delayed(cached_f)(i) for i in [1, 2]))
                assert len(os.listdir(f_cache_directory / 'f')) == 3
                cached_f(3)
            assert len(os.listdir(f_cache_directory / 'f')) == 4
        else:
            assert len(os.listdir(f_cache_directory / 'f')) == 4
            cached_f(3)
            assert len(os.listdir(f_cache_directory / 'f')) == 4