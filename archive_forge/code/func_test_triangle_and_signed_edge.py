import pytest
import networkx as nx
def test_triangle_and_signed_edge(self):
    G = nx.cycle_graph(3)
    G.add_edge(0, 1, weight=-1)
    G.add_edge(3, 0, weight=0)
    assert nx.clustering(G)[0] == 1 / 3
    assert nx.clustering(G, weight='weight')[0] == -1 / 3