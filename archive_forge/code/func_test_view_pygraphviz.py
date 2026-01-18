import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_view_pygraphviz(self):
    G = nx.Graph()
    pytest.raises(nx.NetworkXException, nx.nx_agraph.view_pygraphviz, G)
    G = nx.barbell_graph(4, 6)
    nx.nx_agraph.view_pygraphviz(G, show=False)