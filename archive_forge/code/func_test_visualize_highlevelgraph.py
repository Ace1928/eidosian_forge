from __future__ import annotations
import dataclasses
import inspect
import os
import subprocess
import sys
import time
from collections import OrderedDict
from concurrent.futures import Executor
from operator import add, mul
from typing import NamedTuple
import pytest
from tlz import merge, partial
import dask
import dask.bag as db
from dask.base import (
from dask.delayed import Delayed, delayed
from dask.diagnostics import Profiler
from dask.highlevelgraph import HighLevelGraph
from dask.utils import tmpdir, tmpfile
from dask.utils_test import dec, import_or_none, inc
@pytest.mark.skipif('not da')
@pytest.mark.skipif(bool(sys.flags.optimize), reason='graphviz exception with Python -OO flag')
def test_visualize_highlevelgraph():
    graphviz = pytest.importorskip('graphviz')
    with tmpdir() as d:
        x = da.arange(5, chunks=2)
        viz = x.dask.visualize(filename=os.path.join(d, 'mydask.png'))
        assert isinstance(viz, graphviz.Digraph)