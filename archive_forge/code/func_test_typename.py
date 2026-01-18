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
def test_typename():
    assert typename(HighLevelGraph) == 'dask.highlevelgraph.HighLevelGraph'
    assert typename(HighLevelGraph, short=True) == 'dask.HighLevelGraph'