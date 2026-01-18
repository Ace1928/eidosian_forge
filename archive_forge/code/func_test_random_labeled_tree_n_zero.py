import random
import pytest
import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal
def test_random_labeled_tree_n_zero():
    """Tests if n = 0 then the NetworkXPointlessConcept exception is raised."""
    with pytest.raises(nx.NetworkXPointlessConcept):
        T = nx.random_labeled_tree(0, seed=1234)
    with pytest.raises(nx.NetworkXPointlessConcept):
        T = nx.random_labeled_rooted_tree(0, seed=1234)