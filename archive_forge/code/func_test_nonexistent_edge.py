import pytest
import networkx as nx
from networkx.utils import arbitrary_element, edges_equal, nodes_equal
def test_nonexistent_edge():
    """Tests that attempting to contract a nonexistent edge raises an
    exception.

    """
    with pytest.raises(ValueError):
        G = nx.cycle_graph(4)
        nx.contracted_edge(G, (0, 2))