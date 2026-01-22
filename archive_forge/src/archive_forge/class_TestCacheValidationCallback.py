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
class TestCacheValidationCallback:
    """Tests on parameter `cache_validation_callback`"""

    @pytest.fixture()
    def memory(self, tmp_path):
        mem = Memory(location=tmp_path)
        yield mem
        mem.clear()

    def foo(self, x, d, delay=None):
        d['run'] = True
        if delay is not None:
            time.sleep(delay)
        return x * 2

    def test_invalid_cache_validation_callback(self, memory):
        """Test invalid values for `cache_validation_callback"""
        match = 'cache_validation_callback needs to be callable. Got True.'
        with pytest.raises(ValueError, match=match):
            memory.cache(cache_validation_callback=True)

    @pytest.mark.parametrize('consider_cache_valid', [True, False])
    def test_constant_cache_validation_callback(self, memory, consider_cache_valid):
        """Test expiry of old results"""
        f = memory.cache(self.foo, cache_validation_callback=lambda _: consider_cache_valid, ignore=['d'])
        d1, d2 = ({'run': False}, {'run': False})
        assert f(2, d1) == 4
        assert f(2, d2) == 4
        assert d1['run']
        assert d2['run'] != consider_cache_valid

    def test_memory_only_cache_long_run(self, memory):
        """Test cache validity based on run duration."""

        def cache_validation_callback(metadata):
            duration = metadata['duration']
            if duration > 0.1:
                return True
        f = memory.cache(self.foo, cache_validation_callback=cache_validation_callback, ignore=['d'])
        d1, d2 = ({'run': False}, {'run': False})
        assert f(2, d1, delay=0) == 4
        assert f(2, d2, delay=0) == 4
        assert d1['run']
        assert d2['run']
        d1, d2 = ({'run': False}, {'run': False})
        assert f(2, d1, delay=0.2) == 4
        assert f(2, d2, delay=0.2) == 4
        assert d1['run']
        assert not d2['run']

    def test_memory_expires_after(self, memory):
        """Test expiry of old cached results"""
        f = memory.cache(self.foo, cache_validation_callback=expires_after(seconds=0.3), ignore=['d'])
        d1, d2, d3 = ({'run': False}, {'run': False}, {'run': False})
        assert f(2, d1) == 4
        assert f(2, d2) == 4
        time.sleep(0.5)
        assert f(2, d3) == 4
        assert d1['run']
        assert not d2['run']
        assert d3['run']