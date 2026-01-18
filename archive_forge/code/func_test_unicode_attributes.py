import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_unicode_attributes(self):
    G = nx.Graph()
    name1 = chr(2344) + chr(123) + chr(6543)
    name2 = chr(5543) + chr(1543) + chr(324)
    node_type = str
    G.add_edge(name1, 'Radiohead', foo=name2)
    fd, fname = tempfile.mkstemp()
    self.writer(G, fname)
    H = nx.read_graphml(fname, node_type=node_type)
    assert G._adj == H._adj
    os.close(fd)
    os.unlink(fname)