import pytest
import networkx as nx
def test_odd_degree_sum(self):
    """Tests that a degree sequence whose sum is odd yields an
        exception.

        """
    with pytest.raises(nx.NetworkXError):
        nx.configuration_model([1, 2])