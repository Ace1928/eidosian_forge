import pytest
import networkx as nx
from networkx.utils import py_random_state
def test_spanner_weighted_complete_graph():
    """Test spanner construction on a complete weighted graph."""
    G = nx.complete_graph(20)
    _assign_random_weights(G, seed=_seed)
    spanner = nx.spanner(G, 4, weight='weight', seed=_seed)
    _test_spanner(G, spanner, 4, weight='weight')
    spanner = nx.spanner(G, 10, weight='weight', seed=_seed)
    _test_spanner(G, spanner, 10, weight='weight')