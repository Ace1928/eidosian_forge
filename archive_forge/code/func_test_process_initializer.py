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
@pytest.mark.parametrize('scheduler, initializer, expected_results', [('threading', None, [1] * 10), ('processes', None, [0] * 10), ('processes', proc_init, [1] * 10)])
def test_process_initializer(scheduler, initializer, expected_results):

    @delayed(pure=False)
    def f():
        return global_.value
    global_.value = 1
    with dask.config.set({'scheduler': scheduler, 'multiprocessing.initializer': initializer}):
        results, = compute([f() for _ in range(10)])
    assert results == expected_results
    results2, = compute([f() for _ in range(10)], scheduler=scheduler, initializer=initializer)
    assert results2 == expected_results