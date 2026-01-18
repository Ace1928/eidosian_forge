import pytest
import networkx as nx
def test_null_tree(self):
    with pytest.raises(nx.NetworkXPointlessConcept):
        nx.is_tree(self.graph())