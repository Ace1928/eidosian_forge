import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_number_of_nodes(self):
    G = nx.thresholded_random_geometric_graph(50, 0.2, 0.1, seed=42)
    assert len(G) == 50
    G = nx.thresholded_random_geometric_graph(range(50), 0.2, 0.1, seed=42)
    assert len(G) == 50