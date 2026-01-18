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
@pytest.mark.parametrize('annotation', [{'worker': 'alice'}, {'block_id': annot_map_fn}])
def test_single_annotation(annotation):
    da = pytest.importorskip('dask.array')
    with dask.annotate(**annotation):
        A = da.ones((10, 10), chunks=(5, 5))
    alayer = A.__dask_graph__().layers[A.name]
    assert alayer.annotations == annotation
    assert not dask.get_annotations()