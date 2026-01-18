import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_odd_triangles_error(self):
    G = nx.diamond_graph()
    pytest.raises(nx.NetworkXError, line._odd_triangle, G, (0, 1, 4))
    pytest.raises(nx.NetworkXError, line._odd_triangle, G, (0, 1, 3))