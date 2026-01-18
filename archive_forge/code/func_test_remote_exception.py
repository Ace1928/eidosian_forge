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
def test_remote_exception():
    e = TypeError('hello')
    a = remote_exception(e, 'traceback-body')
    b = remote_exception(e, 'traceback-body')
    assert type(a) == type(b)
    assert isinstance(a, TypeError)
    assert 'hello' in str(a)
    assert 'Traceback' in str(a)
    assert 'traceback-body' in str(a)