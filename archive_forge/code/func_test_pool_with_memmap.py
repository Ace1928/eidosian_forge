import os
import mmap
import sys
import platform
import gc
import pickle
import itertools
from time import sleep
import subprocess
import threading
import faulthandler
import pytest
from joblib.test.common import with_numpy, np
from joblib.test.common import with_multiprocessing
from joblib.test.common import with_dev_shm
from joblib.testing import raises, parametrize, skipif
from joblib.backports import make_memmap
from joblib.parallel import Parallel, delayed
from joblib.pool import MemmappingPool
from joblib.executor import _TestingMemmappingExecutor as TestExecutor
from joblib._memmapping_reducer import has_shareable_memory
from joblib._memmapping_reducer import ArrayMemmapForwardReducer
from joblib._memmapping_reducer import _strided_from_memmap
from joblib._memmapping_reducer import _get_temp_dir
from joblib._memmapping_reducer import _WeakArrayKeyMap
from joblib._memmapping_reducer import _get_backing_memmap
import joblib._memmapping_reducer as jmr
@with_numpy
@with_multiprocessing
@parametrize('factory', [MemmappingPool, TestExecutor.get_memmapping_executor], ids=['multiprocessing', 'loky'])
def test_pool_with_memmap(factory, tmpdir):
    """Check that subprocess can access and update shared memory memmap"""
    assert_array_equal = np.testing.assert_array_equal
    pool_temp_folder = tmpdir.mkdir('pool').strpath
    p = factory(10, max_nbytes=2, temp_folder=pool_temp_folder)
    try:
        filename = tmpdir.join('test.mmap').strpath
        a = np.memmap(filename, dtype=np.float32, shape=(3, 5), mode='w+')
        a.fill(1.0)
        p.map(inplace_double, [(a, (i, j), 1.0) for i in range(a.shape[0]) for j in range(a.shape[1])])
        assert_array_equal(a, 2 * np.ones(a.shape))
        b = np.memmap(filename, dtype=np.float32, shape=(5, 3), mode='c')
        p.map(inplace_double, [(b, (i, j), 2.0) for i in range(b.shape[0]) for j in range(b.shape[1])])
        assert os.listdir(pool_temp_folder) == []
        assert_array_equal(a, 2 * np.ones(a.shape))
        assert_array_equal(b, 2 * np.ones(b.shape))
        c = np.memmap(filename, dtype=np.float32, shape=(10,), mode='r', offset=5 * 4)
        with raises(AssertionError):
            p.map(check_array, [(c, i, 3.0) for i in range(c.shape[0])])
        with raises((RuntimeError, ValueError)):
            p.map(inplace_double, [(c, i, 2.0) for i in range(c.shape[0])])
    finally:
        p.terminate()
        del p