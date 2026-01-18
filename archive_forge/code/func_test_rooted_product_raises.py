import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_rooted_product_raises():
    with pytest.raises(nx.NetworkXError):
        nx.rooted_product(nx.Graph(), nx.path_graph(2), 10)