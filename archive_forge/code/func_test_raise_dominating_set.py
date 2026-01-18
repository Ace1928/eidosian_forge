import pytest
import networkx as nx
def test_raise_dominating_set():
    with pytest.raises(nx.NetworkXError):
        G = nx.path_graph(4)
        D = nx.dominating_set(G, start_with=10)