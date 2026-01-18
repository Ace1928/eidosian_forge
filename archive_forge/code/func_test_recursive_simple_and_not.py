from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_recursive_simple_and_not(self):
    for k in range(2, 10):
        G = self.worst_case_graph(k)
        cc = sorted(nx.simple_cycles(G))
        rcc = sorted(nx.recursive_simple_cycles(G))
        assert len(cc) == len(rcc)
        for c in cc:
            assert any((self.is_cyclic_permutation(c, r) for r in rcc))
        for rc in rcc:
            assert any((self.is_cyclic_permutation(rc, c) for c in cc))