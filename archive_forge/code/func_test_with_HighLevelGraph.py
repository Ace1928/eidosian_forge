from __future__ import annotations
import contextlib
import importlib
import time
from typing import TYPE_CHECKING
def test_with_HighLevelGraph(self):
    from dask.highlevelgraph import HighLevelGraph
    layers = {'a': {'x': 1, 'y': (inc, 'x')}, 'b': {'z': (add, (inc, 'x'), 'y')}}
    dependencies = {'a': (), 'b': {'a'}}
    graph = HighLevelGraph(layers, dependencies)
    assert self.get(graph, 'z') == 4