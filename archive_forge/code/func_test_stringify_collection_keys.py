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
def test_stringify_collection_keys():
    obj = 'Hello'
    assert stringify_collection_keys(obj) is obj
    obj = [('a', 0), (b'a', 0), (1, 1)]
    res = stringify_collection_keys(obj)
    assert res[0] == str(obj[0])
    assert res[1] == str(obj[1])
    assert res[2] == obj[2]