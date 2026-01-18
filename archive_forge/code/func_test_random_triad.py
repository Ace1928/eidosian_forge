import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_random_triad():
    """Tests the random_triad function"""
    G = nx.karate_club_graph()
    G = G.to_directed()
    for i in range(100):
        assert nx.is_triad(nx.random_triad(G))
    G = nx.DiGraph()
    msg = 'at least 3 nodes to form a triad'
    with pytest.raises(nx.NetworkXError, match=msg):
        nx.random_triad(G)