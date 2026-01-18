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
@pytest.mark.parametrize('random', ['numpy', 'random'])
def test_random_seeds(random):
    if random == 'numpy':
        np = pytest.importorskip('numpy')
        random = np.random
    else:
        import random

    @delayed(pure=False)
    def f():
        return tuple((random.randint(0, 10000) for i in range(5)))
    N = 10
    with dask.config.set(scheduler='processes'):
        results, = compute([f() for _ in range(N)])
    assert len(set(results)) == N