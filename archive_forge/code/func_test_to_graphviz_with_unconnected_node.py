from __future__ import annotations
import copy
import os
import re
import sys
from functools import partial
from operator import add, neg
import pytest
from dask.dot import _to_cytoscape_json, cytoscape_graph
from dask import delayed
from dask.utils import ensure_not_exists
def test_to_graphviz_with_unconnected_node():
    dsk2 = dsk.copy()
    dsk2['g'] = 3
    g = to_graphviz(dsk2, verbose=True)
    labels = list(filter(None, map(get_label, g.body)))
    assert len(labels) == 11
    assert set(labels) == {'a', 'b', 'c', 'd', 'e', 'f', 'g'}
    g = to_graphviz(dsk, verbose=True, collapse_outputs=True)
    labels = list(filter(None, map(get_label, g.body)))
    assert len(labels) == 6
    assert set(labels) == {'a', 'b', 'c', 'd', 'e', 'f'}