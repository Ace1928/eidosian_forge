import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.slow
def test_random_spanning_tree_additive_large():
    """
    Sample many spanning trees from the additive method.
    """
    from random import Random
    pytest.importorskip('numpy')
    stats = pytest.importorskip('scipy.stats')
    edges = {(0, 1): 1, (0, 2): 1, (0, 5): 3, (1, 2): 2, (1, 4): 3, (2, 3): 3, (5, 3): 4, (5, 4): 5, (4, 3): 4}
    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v, weight=edges[u, v])
    total_weight = 0
    tree_expected = {}
    for t in nx.SpanningTreeIterator(G):
        weight = 0
        for u, v, d in t.edges(data='weight'):
            weight += d
        tree_expected[t] = weight
        total_weight += weight
    assert len(tree_expected) == 75
    sample_size = 500
    tree_actual = {}
    for t in tree_expected:
        tree_expected[t] = tree_expected[t] / total_weight * sample_size
        tree_actual[t] = 0
    rng = Random(37)
    for _ in range(sample_size):
        sampled_tree = nx.random_spanning_tree(G, 'weight', multiplicative=False, seed=rng)
        assert nx.is_tree(sampled_tree)
        for t in tree_expected:
            if nx.utils.edges_equal(t.edges, sampled_tree.edges):
                tree_actual[t] += 1
                break
    _, p = stats.chisquare(list(tree_actual.values()), list(tree_expected.values()))
    assert not p < 0.05