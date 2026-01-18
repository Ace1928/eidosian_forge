import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_view_pygraphviz_multigraph_edgelabels(self):
    G = nx.MultiGraph()
    G.add_edge(0, 1, key=0, name='left_fork')
    G.add_edge(0, 1, key=1, name='right_fork')
    path, A = nx.nx_agraph.view_pygraphviz(G, edgelabel='name', show=False)
    edges = A.edges()
    assert len(edges) == 2
    for edge in edges:
        assert edge.attr['label'].strip() in ('left_fork', 'right_fork')