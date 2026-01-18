import random
import pytest
import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal
def test_random_unlabeled_forest_n_zero():
    """Tests generation of empty unlabeled forests."""
    F = nx.random_unlabeled_rooted_forest(0, seed=1234)
    assert len(F) == 0
    assert len(F.graph['roots']) == 0