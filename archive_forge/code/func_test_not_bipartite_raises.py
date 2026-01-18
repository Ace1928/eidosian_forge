import pytest
import networkx as nx
def test_not_bipartite_raises():
    G = nx.complete_graph(5)
    with pytest.raises(nx.NetworkXError, match='.*not bipartite'):
        nx.bipartite.maximal_extendability(G)