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
def test_pool_with_memmap_array_view(factory, tmpdir):
    """Check that subprocess can access and update shared memory array"""
    assert_array_equal = np.testing.assert_array_equal
    pool_temp_folder = tmpdir.mkdir('pool').strpath
    p = factory(10, max_nbytes=2, temp_folder=pool_temp_folder)
    try:
        filename = tmpdir.join('test.mmap').strpath
        a = np.memmap(filename, dtype=np.float32, shape=(3, 5), mode='w+')
        a.fill(1.0)
        a_view = np.asarray(a)
        assert not isinstance(a_view, np.memmap)
        assert has_shareable_memory(a_view)
        p.map(inplace_double, [(a_view, (i, j), 1.0) for i in range(a.shape[0]) for j in range(a.shape[1])])
        assert_array_equal(a, 2 * np.ones(a.shape))
        assert_array_equal(a_view, 2 * np.ones(a.shape))
        assert os.listdir(pool_temp_folder) == []
    finally:
        p.terminate()
        del p