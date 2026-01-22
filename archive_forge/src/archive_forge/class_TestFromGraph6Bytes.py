import tempfile
from io import BytesIO
import pytest
import networkx as nx
import networkx.readwrite.graph6 as g6
from networkx.utils import edges_equal, nodes_equal
class TestFromGraph6Bytes:

    def test_from_graph6_bytes(self):
        data = b'DF{'
        G = nx.from_graph6_bytes(data)
        assert nodes_equal(G.nodes(), [0, 1, 2, 3, 4])
        assert edges_equal(G.edges(), [(0, 3), (0, 4), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)])

    def test_read_equals_from_bytes(self):
        data = b'DF{'
        G = nx.from_graph6_bytes(data)
        fh = BytesIO(data)
        Gin = nx.read_graph6(fh)
        assert nodes_equal(G.nodes(), Gin.nodes())
        assert edges_equal(G.edges(), Gin.edges())