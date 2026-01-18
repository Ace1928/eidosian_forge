import pytest
import networkx as nx
def test_richclub():
    G = nx.Graph([(0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (4, 5)])
    rc = nx.richclub.rich_club_coefficient(G, normalized=False)
    assert rc == {0: 12.0 / 30, 1: 8.0 / 12}
    rc0 = nx.richclub.rich_club_coefficient(G, normalized=False)[0]
    assert rc0 == 12.0 / 30.0