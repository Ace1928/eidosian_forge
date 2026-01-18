import random
import networkx as nx
from networkx.algorithms.approximation import maxcut
def test_random_partitioning_all_to_one():
    G = nx.complete_graph(5)
    _, (set1, set2) = maxcut.randomized_partitioning(G, p=1)
    _is_valid_cut(G, set1, set2)
    assert len(set1) == G.number_of_nodes()
    assert len(set2) == 0