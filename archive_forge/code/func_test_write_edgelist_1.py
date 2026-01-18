import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_write_edgelist_1(self):
    fh = io.BytesIO()
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3)])
    nx.write_edgelist(G, fh, data=False)
    fh.seek(0)
    assert fh.read() == b'1 2\n2 3\n'