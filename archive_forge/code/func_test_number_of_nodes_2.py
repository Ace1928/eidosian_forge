import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_number_of_nodes_2(self):
    G = nx.waxman_graph(50, 0.5, 0.1, L=1)
    assert len(G) == 50
    G = nx.waxman_graph(range(50), 0.5, 0.1, L=1)
    assert len(G) == 50