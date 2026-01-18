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
def test_pure_global_setting():
    func = delayed(add)
    with dask.config.set(delayed_pure=True):
        assert func(1, 2).key == func(1, 2).key
    with dask.config.set(delayed_pure=False):
        assert func(1, 2).key != func(1, 2).key
    func = delayed(add, pure=True)
    with dask.config.set(delayed_pure=False):
        assert func(1, 2).key == func(1, 2).key
    assert delayed(1).key != delayed(1).key
    with dask.config.set(delayed_pure=True):
        assert delayed(1).key == delayed(1).key
    with dask.config.set(delayed_pure=False):
        assert delayed(1, pure=True).key == delayed(1, pure=True).key
    data = delayed([1, 2, 3])
    assert data.index(1).key != data.index(1).key
    with dask.config.set(delayed_pure=True):
        assert data.index(1).key == data.index(1).key
        assert data.index(1, pure=False).key != data.index(1, pure=False).key
    with dask.config.set(delayed_pure=False):
        assert data.index(1, pure=True).key == data.index(1, pure=True).key
    with dask.config.set(delayed_pure=False):
        assert data.index.key == data.index.key
        element = data[0]
        assert (element + element).key == (element + element).key