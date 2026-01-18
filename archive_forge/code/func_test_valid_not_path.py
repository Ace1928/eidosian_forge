import math
from itertools import permutations
from pytest import raises
import networkx as nx
from networkx.algorithms.matching import matching_dict_to_set
from networkx.utils import edges_equal
def test_valid_not_path(self):
    G = nx.cycle_graph(4)
    G.add_edge(0, 4)
    G.add_edge(1, 4)
    G.add_edge(5, 2)
    assert nx.is_perfect_matching(G, {(1, 4), (0, 3), (5, 2)})