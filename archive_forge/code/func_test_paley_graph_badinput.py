import pytest
import networkx as nx
def test_paley_graph_badinput():
    with pytest.raises(nx.NetworkXError, match='`create_using` cannot be a multigraph.'):
        nx.paley_graph(3, create_using=nx.MultiGraph)