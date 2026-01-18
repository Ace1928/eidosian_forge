import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_theta(self):
    """Tests that pairs of vertices adjacent if and only if their sum
        weights exceeds the threshold parameter theta.
        """
    G = nx.thresholded_random_geometric_graph(50, 0.25, 0.1, seed=42)
    for u, v in combinations(G, 2):
        if v in G[u]:
            assert G.nodes[u]['weight'] + G.nodes[v]['weight'] >= 0.1