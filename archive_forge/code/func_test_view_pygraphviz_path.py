import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_view_pygraphviz_path(self, tmp_path):
    G = nx.complete_graph(3)
    input_path = str(tmp_path / 'graph.png')
    out_path, A = nx.nx_agraph.view_pygraphviz(G, path=input_path, show=False)
    assert out_path == input_path
    with open(input_path, 'rb') as fh:
        data = fh.read()
    assert len(data) > 0