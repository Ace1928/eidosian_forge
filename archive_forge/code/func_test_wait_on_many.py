from __future__ import annotations
import random
import time
from operator import add
import pytest
import dask
import dask.bag as db
from dask import delayed
from dask.base import clone_key
from dask.blockwise import Blockwise
from dask.graph_manipulation import bind, checkpoint, chunks, clone, wait_on
from dask.highlevelgraph import HighLevelGraph
from dask.tests.test_base import Tuple
from dask.utils_test import import_or_none
@pytest.mark.parametrize('layers', [False, True])
def test_wait_on_many(layers):
    t1, t2, cnt = demo_tuples(layers)
    out = wait_on(t1, {'x': [t2]})
    assert dask.compute(*out, scheduler='sync') == ((1, 2, 3), {'x': [(4, 5)]})
    assert cnt.n == 5