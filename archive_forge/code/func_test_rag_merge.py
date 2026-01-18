import pytest
from numpy.testing import assert_array_equal
import numpy as np
from skimage import graph
from skimage import segmentation, data
from skimage._shared import testing
def test_rag_merge():
    g = graph.RAG()
    for i in range(5):
        g.add_node(i, {'labels': [i]})
    g.add_edge(0, 1, {'weight': 10})
    g.add_edge(1, 2, {'weight': 20})
    g.add_edge(2, 3, {'weight': 30})
    g.add_edge(3, 0, {'weight': 40})
    g.add_edge(0, 2, {'weight': 50})
    g.add_edge(3, 4, {'weight': 60})
    gc = g.copy()
    g.merge_nodes(0, 2)
    assert g.adj[1][2]['weight'] == 10
    assert g.adj[2][3]['weight'] == 30
    gc.merge_nodes(0, 2, weight_func=max_edge)
    assert gc.adj[1][2]['weight'] == 20
    assert gc.adj[2][3]['weight'] == 40
    g.merge_nodes(1, 4)
    g.merge_nodes(2, 3)
    n = g.merge_nodes(3, 4, in_place=False)
    assert sorted(g.nodes[n]['labels']) == list(range(5))
    assert list(g.edges()) == []