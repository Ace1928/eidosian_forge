import random
import pytest
import networkx as nx
from networkx.utils import arbitrary_element, graphs_equal
@pytest.mark.filterwarnings('ignore')
def test_random_directed_tree():
    """Generates a directed tree."""
    T = nx.random_tree(10, seed=1234, create_using=nx.DiGraph())
    assert T.is_directed()