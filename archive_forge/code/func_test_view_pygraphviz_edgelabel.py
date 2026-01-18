import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_view_pygraphviz_edgelabel(self):
    G = nx.Graph()
    G.add_edge(1, 2, weight=7)
    G.add_edge(2, 3, weight=8)
    path, A = nx.nx_agraph.view_pygraphviz(G, edgelabel='weight', show=False)
    for edge in A.edges():
        assert edge.attr['weight'] in ('7', '8')