from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_simple_cycles_bounded(self):
    d = nx.DiGraph()
    expected = []
    for n in range(10):
        nx.add_cycle(d, range(n))
        expected.append(n)
        for k, e in enumerate(expected):
            self.check_cycle_algorithm(d, e, length_bound=k)
    g = nx.Graph()
    top = 0
    expected = []
    for n in range(10):
        expected.append(n if n < 2 else n - 1)
        if n == 2:
            continue
        nx.add_cycle(g, range(top, top + n))
        top += n
        for k, e in enumerate(expected):
            self.check_cycle_algorithm(g, e, length_bound=k)