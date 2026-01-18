import pytest
import networkx as nx
def test_planted_partition_graph():
    G = nx.planted_partition_graph(4, 3, 1, 0, seed=42)
    C = G.graph['partition']
    assert len(C) == 4
    assert len(G) == 12
    assert len(list(G.edges())) == 12
    G = nx.planted_partition_graph(4, 3, 0, 1)
    C = G.graph['partition']
    assert len(C) == 4
    assert len(G) == 12
    assert len(list(G.edges())) == 54
    G = nx.planted_partition_graph(10, 4, 0.5, 0.1, seed=42)
    C = G.graph['partition']
    assert len(C) == 10
    assert len(G) == 40
    G = nx.planted_partition_graph(4, 3, 1, 0, directed=True)
    C = G.graph['partition']
    assert len(C) == 4
    assert len(G) == 12
    assert len(list(G.edges())) == 24
    G = nx.planted_partition_graph(4, 3, 0, 1, directed=True)
    C = G.graph['partition']
    assert len(C) == 4
    assert len(G) == 12
    assert len(list(G.edges())) == 108
    G = nx.planted_partition_graph(10, 4, 0.5, 0.1, seed=42, directed=True)
    C = G.graph['partition']
    assert len(C) == 10
    assert len(G) == 40
    ppg = nx.planted_partition_graph
    pytest.raises(nx.NetworkXError, ppg, 3, 3, 1.1, 0.1)
    pytest.raises(nx.NetworkXError, ppg, 3, 3, -0.1, 0.1)
    pytest.raises(nx.NetworkXError, ppg, 3, 3, 0.1, 1.1)
    pytest.raises(nx.NetworkXError, ppg, 3, 3, 0.1, -0.1)