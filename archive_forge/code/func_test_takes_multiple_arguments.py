from __future__ import annotations
import datetime
import functools
import operator
import pickle
from array import array
import pytest
from tlz import curry
from dask import get
from dask.highlevelgraph import HighLevelGraph
from dask.optimization import SubgraphCallable
from dask.utils import (
from dask.utils_test import inc
def test_takes_multiple_arguments():
    assert takes_multiple_arguments(map)
    assert not takes_multiple_arguments(sum)

    def multi(a, b, c):
        return (a, b, c)

    class Singular:

        def __init__(self, a):
            pass

    class Multi:

        def __init__(self, a, b):
            pass
    assert takes_multiple_arguments(multi)
    assert not takes_multiple_arguments(Singular)
    assert takes_multiple_arguments(Multi)

    def f():
        pass
    assert not takes_multiple_arguments(f)

    def vararg(*args):
        pass
    assert takes_multiple_arguments(vararg)
    assert not takes_multiple_arguments(vararg, varargs=False)