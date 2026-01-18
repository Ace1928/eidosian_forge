import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.parametrize('graph_type', (nx.Graph, nx.DiGraph, nx.MultiGraph, nx.MultiDiGraph))
def test_set_node_attributes(graph_type):
    G = nx.path_graph(3, create_using=graph_type)
    vals = 100
    attr = 'hello'
    nx.set_node_attributes(G, vals, attr)
    assert G.nodes[0][attr] == vals
    assert G.nodes[1][attr] == vals
    assert G.nodes[2][attr] == vals
    G = nx.path_graph(3, create_using=graph_type)
    vals = dict(zip(sorted(G.nodes()), range(len(G))))
    attr = 'hi'
    nx.set_node_attributes(G, vals, attr)
    assert G.nodes[0][attr] == 0
    assert G.nodes[1][attr] == 1
    assert G.nodes[2][attr] == 2
    G = nx.path_graph(3, create_using=graph_type)
    d = {'hi': 0, 'hello': 200}
    vals = dict.fromkeys(G.nodes(), d)
    vals.pop(0)
    nx.set_node_attributes(G, vals)
    assert G.nodes[0] == {}
    assert G.nodes[1]['hi'] == 0
    assert G.nodes[2]['hello'] == 200