import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_p_dist_zero(self):
    """Tests if p_dict = 0 returns disconnected graph with 0 edges"""

    def p_dist(dist):
        return 0
    G = nx.geographical_threshold_graph(50, 1, p_dist=p_dist)
    assert len(G.edges) == 0