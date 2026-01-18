from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_weighted_diamond(self):
    mcb = nx.minimum_cycle_basis(self.diamond_graph, weight='weight')
    assert_basis_equal(mcb, [[2, 4, 1], [4, 3, 2, 1]])