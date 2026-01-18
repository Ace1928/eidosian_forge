from __future__ import annotations
import multiprocessing
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor
from operator import add
import pytest
import dask
from dask import compute, delayed
from dask.multiprocessing import _dumps, _loads, get, get_context, remote_exception
from dask.system import CPU_COUNT
from dask.utils_test import inc
def test_pickle_locals():
    """Unrelated locals should not be included in serialized bytes"""
    np = pytest.importorskip('numpy')

    def unrelated_function_local(a):
        return np.array([a])

    def my_small_function_local(a, b):
        return a + b
    b = _dumps(my_small_function_local)
    assert b'my_small_function_global' not in b
    assert b'my_small_function_local' in b
    assert b'unrelated_function_local' not in b