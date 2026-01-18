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
def test_pickle_globals():
    """Unrelated globals should not be included in serialized bytes"""
    b = _dumps(my_small_function_global)
    assert b'my_small_function_global' in b
    assert b'unrelated_function_global' not in b
    assert b'numpy' not in b