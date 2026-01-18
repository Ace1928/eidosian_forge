import random
import pytest
import networkx as nx
import networkx.algorithms.approximation as nx_app
def test_spanning_tree_distribution():
    """
    Test that we can create an exponential distribution of spanning trees such
    that the probability of each tree is proportional to the product of edge
    weights.

    Results of this test have been confirmed with hypothesis testing from the
    created distribution.

    This test uses the symmetric, fractional Held Karp solution.
    """
    import networkx.algorithms.approximation.traveling_salesman as tsp
    pytest.importorskip('numpy')
    pytest.importorskip('scipy')
    z_star = {(0, 1): 5 / 12, (0, 2): 5 / 12, (0, 5): 5 / 6, (1, 0): 5 / 12, (1, 2): 1 / 3, (1, 4): 5 / 6, (2, 0): 5 / 12, (2, 1): 1 / 3, (2, 3): 5 / 6, (3, 2): 5 / 6, (3, 4): 1 / 3, (3, 5): 1 / 2, (4, 1): 5 / 6, (4, 3): 1 / 3, (4, 5): 1 / 2, (5, 0): 5 / 6, (5, 3): 1 / 2, (5, 4): 1 / 2}
    solution_gamma = {(0, 1): -0.6383, (0, 2): -0.6827, (0, 5): 0, (1, 2): -1.0781, (1, 4): 0, (2, 3): 0, (5, 3): -0.282, (5, 4): -0.3327, (4, 3): -0.9927}
    G = nx.MultiGraph()
    for u, v in z_star:
        if (u, v) in G.edges or (v, u) in G.edges:
            continue
        G.add_edge(u, v)
    gamma = tsp.spanning_tree_distribution(G, z_star)
    assert {key: round(gamma[key], 4) for key in gamma} == solution_gamma