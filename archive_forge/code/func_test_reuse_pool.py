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
@pytest.mark.parametrize('pool_typ', [multiprocessing.Pool, ProcessPoolExecutor])
def test_reuse_pool(pool_typ):
    with pool_typ(CPU_COUNT) as pool:
        with dask.config.set(pool=pool):
            assert get({'x': (inc, 1)}, 'x') == 2
            assert get({'x': (inc, 1)}, 'x') == 2