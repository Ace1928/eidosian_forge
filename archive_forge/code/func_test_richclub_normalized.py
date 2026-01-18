import pytest
import networkx as nx
def test_richclub_normalized():
    G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (4, 5)])
    rcNorm = nx.richclub.rich_club_coefficient(G, Q=2)
    assert rcNorm == {0: 1.0, 1: 1.0}