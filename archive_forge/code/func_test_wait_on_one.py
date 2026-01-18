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
def test_wait_on_one(layers):
    t1, _, cnt = demo_tuples(layers)
    t1w = wait_on(t1)
    assert t1w.compute(scheduler='sync') == (1, 2, 3)
    assert cnt.n == 3