from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_simple_cycles_graph(self):
    testG = nx.cycle_graph(8)
    cyc1 = tuple(range(8))
    self.check_cycle_algorithm(testG, [cyc1])
    testG.add_edge(4, -1)
    nx.add_path(testG, [3, -2, -3, -4])
    self.check_cycle_algorithm(testG, [cyc1])
    testG.update(nx.cycle_graph(range(8, 16)))
    cyc2 = tuple(range(8, 16))
    self.check_cycle_algorithm(testG, [cyc1, cyc2])
    testG.update(nx.cycle_graph(range(4, 12)))
    cyc3 = tuple(range(4, 12))
    expected = {(0, 1, 2, 3, 4, 5, 6, 7), (8, 9, 10, 11, 12, 13, 14, 15), (4, 5, 6, 7, 8, 9, 10, 11), (4, 5, 6, 7, 8, 15, 14, 13, 12, 11), (0, 1, 2, 3, 4, 11, 10, 9, 8, 7), (0, 1, 2, 3, 4, 11, 12, 13, 14, 15, 8, 7)}
    self.check_cycle_algorithm(testG, expected)
    assert len(expected) == 2 ** 3 - 1 - 1
    testG = nx.cycle_graph(12)
    testG.update(nx.cycle_graph([12, 10, 13, 2, 14, 4, 15, 8]).edges)
    expected = 2 ** 5 - 1 - 11
    self.check_cycle_algorithm(testG, expected)