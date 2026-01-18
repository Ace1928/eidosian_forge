import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_shortest_simple_paths_directed_with_weight_function():

    def cost(u, v, x):
        return 1
    G = cnlti(nx.grid_2d_graph(4, 4), first_label=1, ordering='sorted')
    paths = nx.shortest_simple_paths(G, 1, 12)
    assert next(paths) == [1, 2, 3, 4, 8, 12]
    assert next(paths) == [1, 5, 6, 7, 8, 12]
    assert [len(path) for path in nx.shortest_simple_paths(G, 1, 12, weight=cost)] == sorted((len(path) for path in nx.all_simple_paths(G, 1, 12)))