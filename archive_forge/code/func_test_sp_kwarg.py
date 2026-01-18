from random import Random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.distance_measures import _extrema_bounding
def test_sp_kwarg(self):
    K_5 = nx.complete_graph(5)
    sp = dict(nx.shortest_path_length(K_5))
    assert nx.barycenter(K_5, sp=sp) == list(K_5)
    for u, v, data in K_5.edges.data():
        data['weight'] = 1
    pytest.raises(ValueError, nx.barycenter, K_5, sp=sp, weight='weight')
    del sp[0][1]
    pytest.raises(nx.NetworkXNoPath, nx.barycenter, K_5, sp=sp)