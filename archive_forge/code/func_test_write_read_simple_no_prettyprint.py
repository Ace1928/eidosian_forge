import io
import os
import tempfile
import pytest
import networkx as nx
from networkx.readwrite.graphml import GraphMLWriter
from networkx.utils import edges_equal, nodes_equal
def test_write_read_simple_no_prettyprint(self):
    G = self.simple_directed_graph
    G.graph['hi'] = 'there'
    G.graph['id'] = '1'
    fh = io.BytesIO()
    self.writer(G, fh, prettyprint=False)
    fh.seek(0)
    H = nx.read_graphml(fh)
    assert sorted(G.nodes()) == sorted(H.nodes())
    assert sorted(G.edges()) == sorted(H.edges())
    assert sorted(G.edges(data=True)) == sorted(H.edges(data=True))
    self.simple_directed_fh.seek(0)