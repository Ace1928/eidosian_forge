import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_number_of_nodes_1(self):
    G = nx.waxman_graph(50, 0.5, 0.1, seed=42)
    assert len(G) == 50
    G = nx.waxman_graph(range(50), 0.5, 0.1, seed=42)
    assert len(G) == 50