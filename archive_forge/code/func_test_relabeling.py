import tempfile
from io import BytesIO
import pytest
import networkx as nx
import networkx.readwrite.graph6 as g6
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.parametrize('edge', ((0, 1), (1, 2), (1, 42)))
def test_relabeling(self, edge):
    G = nx.Graph([edge])
    assert g6.to_graph6_bytes(G) == b'>>graph6<<A_\n'