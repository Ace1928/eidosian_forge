import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_read_multiline_adjlist_1(self):
    s = b'# comment line\n1 2\n# comment line\n2\n3\n'
    bytesIO = io.BytesIO(s)
    G = nx.read_multiline_adjlist(bytesIO)
    adj = {'1': {'3': {}, '2': {}}, '3': {'1': {}}, '2': {'1': {}}}
    assert graphs_equal(G, nx.Graph(adj))