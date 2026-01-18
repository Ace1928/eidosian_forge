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
def test_to_graphviz_attributes():
    assert to_graphviz(dsk).graph_attr['rankdir'] == 'BT'
    assert to_graphviz(dsk, rankdir='LR').graph_attr['rankdir'] == 'LR'
    assert to_graphviz(dsk, node_attr={'color': 'white'}).node_attr['color'] == 'white'
    assert to_graphviz(dsk, edge_attr={'color': 'white'}).edge_attr['color'] == 'white'