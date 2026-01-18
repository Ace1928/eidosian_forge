import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.slow
def test_random_spanning_tree_multiplicative_large():
    """
    Sample many trees from the distribution created in the last test
    """
    from math import exp
    from random import Random
    pytest.importorskip('numpy')
    stats = pytest.importorskip('scipy.stats')
    gamma = {(0, 1): -0.6383, (0, 2): -0.6827, (0, 5): 0, (1, 2): -1.0781, (1, 4): 0, (2, 3): 0, (5, 3): -0.282, (5, 4): -0.3327, (4, 3): -0.9927}
    G = nx.Graph()
    for u, v in gamma:
        G.add_edge(u, v, lambda_key=exp(gamma[u, v]))
    total_weight = 0
    tree_expected = {}
    for t in nx.SpanningTreeIterator(G):
        weight = 1
        for u, v, d in t.edges(data='lambda_key'):
            weight *= d
        tree_expected[t] = weight
        total_weight += weight
    assert len(tree_expected) == 75
    sample_size = 1200
    tree_actual = {}
    for t in tree_expected:
        tree_expected[t] = tree_expected[t] / total_weight * sample_size
        tree_actual[t] = 0
    rng = Random(37)
    for _ in range(sample_size):
        sampled_tree = nx.random_spanning_tree(G, 'lambda_key', seed=rng)
        assert nx.is_tree(sampled_tree)
        for t in tree_expected:
            if nx.utils.edges_equal(t.edges, sampled_tree.edges):
                tree_actual[t] += 1
                break
    _, p = stats.chisquare(list(tree_actual.values()), list(tree_expected.values()))
    assert not p < 0.05