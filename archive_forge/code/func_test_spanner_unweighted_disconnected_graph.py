import pytest
import networkx as nx
from networkx.utils import py_random_state
def test_spanner_unweighted_disconnected_graph():
    """Test spanner construction on a disconnected graph."""
    G = nx.disjoint_union(nx.complete_graph(10), nx.complete_graph(10))
    spanner = nx.spanner(G, 4, seed=_seed)
    _test_spanner(G, spanner, 4)
    spanner = nx.spanner(G, 10, seed=_seed)
    _test_spanner(G, spanner, 10)