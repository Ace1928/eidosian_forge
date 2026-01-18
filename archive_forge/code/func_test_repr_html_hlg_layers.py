from __future__ import annotations
import os
import threading
import xml.etree.ElementTree
from collections.abc import Set
from concurrent.futures import ThreadPoolExecutor
import pytest
import dask
from dask.base import tokenize
from dask.blockwise import Blockwise, blockwise_token
from dask.highlevelgraph import HighLevelGraph, Layer, MaterializedLayer, to_graphviz
from dask.utils_test import inc
def test_repr_html_hlg_layers():
    pytest.importorskip('jinja2')
    hg = HighLevelGraph({'a': {'a': 1, ('a', 0): 2, 'b': 3}, 'b': {'c': 4}}, {'a': set(), 'b': set()})
    assert xml.etree.ElementTree.fromstring(hg._repr_html_()) is not None
    for layer in hg.layers.values():
        assert xml.etree.ElementTree.fromstring(layer._repr_html_()) is not None