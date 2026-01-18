import tempfile
from io import BytesIO
import pytest
import networkx as nx
import networkx.readwrite.graph6 as g6
from networkx.utils import edges_equal, nodes_equal
def test_read_equals_from_bytes(self):
    data = b'DF{'
    G = nx.from_graph6_bytes(data)
    fh = BytesIO(data)
    Gin = nx.read_graph6(fh)
    assert nodes_equal(G.nodes(), Gin.nodes())
    assert edges_equal(G.edges(), Gin.edges())