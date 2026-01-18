import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_tadpole_graph_same_as_path_when_m1_is_2_or_0(self):
    for m1, m2 in [(2, 0), (2, 5), (2, 10), ('ab', 20)]:
        G = nx.tadpole_graph(m1, m2)
        assert is_isomorphic(G, nx.path_graph(m2 + 2))