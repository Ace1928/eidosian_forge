import pytest
from numpy.testing import assert_array_equal
import numpy as np
from skimage import graph
from skimage import segmentation, data
from skimage._shared import testing
@pytest.mark.parametrize('in_place', [True, False])
def test_rag_merge_gh5360(in_place):
    g = graph.RAG()
    g.add_edge(1, 2, weight=10)
    g.add_edge(2, 3, weight=20)
    g.add_edge(3, 4, weight=30)
    g.add_edge(4, 1, weight=40)
    g.add_edge(1, 3, weight=50)
    for n in g.nodes():
        g.nodes[n]['labels'] = [n]
    gc = g.copy()
    merged_id = 3 if in_place is True else 5
    g.merge_nodes(1, 3, in_place=in_place)
    assert g.adj[merged_id][2]['weight'] == 10
    assert g.adj[merged_id][4]['weight'] == 30
    gc.merge_nodes(1, 3, weight_func=max_edge, in_place=in_place)
    assert gc.adj[merged_id][2]['weight'] == 20
    assert gc.adj[merged_id][4]['weight'] == 40