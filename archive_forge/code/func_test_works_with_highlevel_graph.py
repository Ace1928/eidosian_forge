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
def test_works_with_highlevel_graph():
    """Previously `dask.multiprocessing.get` would accidentally forward
    `HighLevelGraph` graphs through the dask optimization/scheduling routines,
    resulting in odd errors. One way to trigger this was to have a
    non-indexable object in a task. This is just a smoketest to ensure that
    things work properly even if `HighLevelGraph` objects get passed to
    `dask.multiprocessing.get`. See https://github.com/dask/dask/issues/7190.
    """

    class NoIndex:

        def __init__(self, x):
            self.x = x

        def __getitem__(self, key):
            raise Exception('Oh no!')
    x = delayed(lambda x: x)(NoIndex(1))
    res, = get(x.dask, x.__dask_keys__())
    assert isinstance(res, NoIndex)
    assert res.x == 1