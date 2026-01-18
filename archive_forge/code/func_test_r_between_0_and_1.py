import math
import random
from itertools import combinations
import pytest
import networkx as nx
def test_r_between_0_and_1(self):
    """Smoke test for radius in range [0, 1]"""
    G = nx.navigable_small_world_graph(3, p=1, q=0, r=0.5, dim=2, seed=42)
    expected = nx.grid_2d_graph(3, 3, create_using=nx.DiGraph)
    assert nx.utils.graphs_equal(G, expected)