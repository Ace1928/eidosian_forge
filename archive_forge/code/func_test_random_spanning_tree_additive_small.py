import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_random_spanning_tree_additive_small():
    """
    Sample a single spanning tree from the additive method.
    """
    pytest.importorskip('scipy')
    edges = {(0, 1): 1, (0, 2): 1, (0, 5): 3, (1, 2): 2, (1, 4): 3, (2, 3): 3, (5, 3): 4, (5, 4): 5, (4, 3): 4}
    G = nx.Graph()
    for u, v in edges:
        G.add_edge(u, v, weight=edges[u, v])
    solution_edges = [(0, 2), (1, 2), (2, 3), (3, 4), (3, 5)]
    solution = nx.Graph()
    solution.add_edges_from(solution_edges)
    sampled_tree = nx.random_spanning_tree(G, weight='weight', multiplicative=False, seed=37)
    assert nx.utils.edges_equal(solution.edges, sampled_tree.edges)