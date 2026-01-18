import pytest
import networkx as nx
from networkx.utils import py_random_state
def test_spanner_trivial():
    """Test a trivial spanner with stretch 1."""
    G = nx.complete_graph(20)
    spanner = nx.spanner(G, 1, seed=_seed)
    for u, v in G.edges:
        assert spanner.has_edge(u, v)