from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_simple_cycles(self):
    edges = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 0), (2, 1), (2, 2)]
    G = nx.DiGraph(edges)
    cc = sorted(nx.simple_cycles(G))
    ca = [[0], [0, 1, 2], [0, 2], [1, 2], [2]]
    assert len(cc) == len(ca)
    for c in cc:
        assert any((self.is_cyclic_permutation(c, rc) for rc in ca))