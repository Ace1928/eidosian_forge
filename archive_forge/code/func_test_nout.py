from __future__ import annotations
import pickle
import types
from collections import namedtuple
from dataclasses import dataclass, field
from functools import partial
from operator import add, setitem
from random import random
from typing import NamedTuple
import cloudpickle
import pytest
from tlz import merge
import dask
import dask.bag as db
from dask import compute
from dask.delayed import Delayed, delayed, to_task_dask
from dask.highlevelgraph import HighLevelGraph
from dask.utils_test import inc
def test_nout():
    func = delayed(lambda x: (x, -x), nout=2, pure=True)
    x = func(1)
    assert len(x) == 2
    a, b = x
    assert compute(a, b) == (1, -1)
    assert a._length is None
    assert b._length is None
    pytest.raises(TypeError, lambda: len(a))
    pytest.raises(TypeError, lambda: list(a))
    pytest.raises(ValueError, lambda: delayed(add, nout=-1))
    pytest.raises(ValueError, lambda: delayed(add, nout=True))
    func = delayed(add, nout=None)
    a = func(1)
    assert a._length is None
    pytest.raises(TypeError, lambda: list(a))
    pytest.raises(TypeError, lambda: len(a))
    func = delayed(lambda x: (x,), nout=1, pure=True)
    x = func(1)
    assert len(x) == 1
    a, = x
    assert a.compute() == 1
    assert a._length is None
    pytest.raises(TypeError, lambda: len(a))
    func = delayed(lambda x: tuple(), nout=0, pure=True)
    x = func(1)
    assert len(x) == 0
    assert x.compute() == tuple()