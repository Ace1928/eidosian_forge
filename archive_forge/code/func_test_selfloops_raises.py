import pytest
import networkx as nx
def test_selfloops_raises():
    G = nx.ladder_graph(3)
    G.add_edge(0, 0)
    with pytest.raises(nx.NetworkXError, match='.*not bipartite'):
        nx.bipartite.maximal_extendability(G)