import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_shortest_simple_paths_directed():
    G = nx.cycle_graph(7, create_using=nx.DiGraph())
    paths = nx.shortest_simple_paths(G, 0, 3)
    assert list(paths) == [[0, 1, 2, 3]]