import itertools
import typing
import pytest
import networkx as nx
from networkx.algorithms.isomorphism.isomorph import graph_could_be_isomorphic
from networkx.utils import edges_equal, nodes_equal
def test_trivial_graph(self):
    assert nx.number_of_nodes(nx.trivial_graph()) == 1