import math
import random
from itertools import combinations
import pytest
import networkx as nx
@pytest.mark.parametrize('seed', range(2478, 2578, 10))
def test_r_general_scaling(self, seed):
    """The probability of adding a long-range edge scales with `1 / dist**r`,
        so a navigable_small_world graph created with r < 1 should generally
        result in more edges than a navigable_small_world graph with r >= 1
        (for 0 < q << n).

        N.B. this is probabilistic, so this test may not hold for all seeds."""
    G1 = nx.navigable_small_world_graph(7, q=3, r=0.5, seed=seed)
    G2 = nx.navigable_small_world_graph(7, q=3, r=1, seed=seed)
    G3 = nx.navigable_small_world_graph(7, q=3, r=2, seed=seed)
    assert G1.number_of_edges() > G2.number_of_edges()
    assert G2.number_of_edges() > G3.number_of_edges()