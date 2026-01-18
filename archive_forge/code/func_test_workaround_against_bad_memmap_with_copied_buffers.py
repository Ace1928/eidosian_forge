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
def test_workaround_against_bad_memmap_with_copied_buffers(factory, tmpdir):
    """Check that memmaps with a bad buffer are returned as regular arrays

    Unary operations and ufuncs on memmap instances return a new memmap
    instance with an in-memory buffer (probably a numpy bug).
    """
    assert_array_equal = np.testing.assert_array_equal
    p = factory(3, max_nbytes=10, temp_folder=tmpdir.strpath)
    try:
        a = np.asarray(np.arange(6000).reshape((1000, 2, 3)), order='F')[:, :1, :]
        b = p.apply_async(_worker_multiply, args=(a, 3)).get()
        assert not has_shareable_memory(b)
        assert_array_equal(b, 3 * a)
    finally:
        p.terminate()
        del p