import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_to_agraph_with_nodedata(self):
    G = nx.Graph()
    G.add_node(1, color='red')
    A = nx.nx_agraph.to_agraph(G)
    assert dict(A.nodes()[0].attr) == {'color': 'red'}