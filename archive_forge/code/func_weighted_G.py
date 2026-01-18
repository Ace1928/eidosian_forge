import pytest
import networkx as nx
def weighted_G():
    G = nx.Graph()
    G.add_edge(0, 1, weight=3)
    G.add_edge(0, 2, weight=2)
    G.add_edge(0, 3, weight=6)
    G.add_edge(0, 4, weight=4)
    G.add_edge(1, 3, weight=5)
    G.add_edge(1, 5, weight=5)
    G.add_edge(2, 4, weight=1)
    G.add_edge(3, 4, weight=2)
    G.add_edge(3, 5, weight=1)
    G.add_edge(4, 5, weight=4)
    return G