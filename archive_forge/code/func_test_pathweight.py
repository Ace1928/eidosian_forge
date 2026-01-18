import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_pathweight():
    valid_path = [1, 2, 3]
    invalid_path = [1, 3, 2]
    graphs = [nx.Graph(), nx.DiGraph(), nx.MultiGraph(), nx.MultiDiGraph()]
    edges = [(1, 2, {'cost': 5, 'dist': 6}), (2, 3, {'cost': 3, 'dist': 4}), (1, 2, {'cost': 1, 'dist': 2})]
    for graph in graphs:
        graph.add_edges_from(edges)
        assert nx.path_weight(graph, valid_path, 'cost') == 4
        assert nx.path_weight(graph, valid_path, 'dist') == 6
        pytest.raises(nx.NetworkXNoPath, nx.path_weight, graph, invalid_path, 'cost')