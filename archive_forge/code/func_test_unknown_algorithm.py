import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_unknown_algorithm():
    with pytest.raises(ValueError):
        nx.minimum_spanning_tree(nx.Graph(), algorithm='random')
    with pytest.raises(ValueError, match='random is not a valid choice for an algorithm.'):
        nx.maximum_spanning_edges(nx.Graph(), algorithm='random')