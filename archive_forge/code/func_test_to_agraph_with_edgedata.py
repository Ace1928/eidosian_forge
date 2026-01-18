import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
@pytest.mark.parametrize('graph_class', (nx.Graph, nx.MultiGraph))
def test_to_agraph_with_edgedata(self, graph_class):
    G = graph_class()
    G.add_nodes_from([0, 1])
    G.add_edge(0, 1, color='yellow')
    A = nx.nx_agraph.to_agraph(G)
    assert dict(A.edges()[0].attr) == {'color': 'yellow'}