import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
@pytest.mark.parametrize(('m', 'n'), [('ab', ''), ('ab', 'c'), ('abc', 'defg')])
def test_tadpole_graph_size_node_sequences(self, m, n):
    G = nx.tadpole_graph(m, n)
    assert nx.number_of_nodes(G) == len(m) + len(n)
    assert nx.number_of_edges(G) == len(m) + len(n) - (len(m) == 2)