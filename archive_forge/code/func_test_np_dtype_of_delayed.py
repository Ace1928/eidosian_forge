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
def test_np_dtype_of_delayed():
    np = pytest.importorskip('numpy')
    x = delayed(1)
    with pytest.raises(TypeError):
        np.dtype(x)
    assert delayed(np.array([1], dtype='f8')).dtype.compute() == np.dtype('f8')