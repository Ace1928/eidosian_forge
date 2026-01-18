import pytest
import networkx as nx
from networkx.utils import py_random_state
def test_spanner_unweighted_gnp_graph():
    """Test spanner construction on an unweighted gnp graph."""
    G = nx.gnp_random_graph(20, 0.4, seed=_seed)
    spanner = nx.spanner(G, 4, seed=_seed)
    _test_spanner(G, spanner, 4)
    spanner = nx.spanner(G, 10, seed=_seed)
    _test_spanner(G, spanner, 10)