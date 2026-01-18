from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_simple_cycles_notable_clique_sequences(self):
    g_family = [self.K(n) for n in range(2, 12)]
    expected = [0, 1, 4, 10, 20, 35, 56, 84, 120, 165, 220]
    self.check_cycle_enumeration_integer_sequence(g_family, expected, length_bound=3)

    def triangles(g, **kwargs):
        yield from (c for c in nx.simple_cycles(g, **kwargs) if len(c) == 3)
    g_family = [self.D(n) for n in range(2, 12)]
    expected = [2 * e for e in expected]
    self.check_cycle_enumeration_integer_sequence(g_family, expected, length_bound=3, algorithm=triangles)

    def four_cycles(g, **kwargs):
        yield from (c for c in nx.simple_cycles(g, **kwargs) if len(c) == 4)
    expected = [0, 0, 0, 3, 15, 45, 105, 210, 378, 630, 990]
    g_family = [self.K(n) for n in range(1, 12)]
    self.check_cycle_enumeration_integer_sequence(g_family, expected, length_bound=4, algorithm=four_cycles)
    expected = [2 * e for e in expected]
    g_family = [self.D(n) for n in range(1, 15)]
    self.check_cycle_enumeration_integer_sequence(g_family, expected, length_bound=4, algorithm=four_cycles)
    expected = [0, 1, 5, 20, 84, 409, 2365]
    g_family = [self.D(n) for n in range(1, 8)]
    self.check_cycle_enumeration_integer_sequence(g_family, expected)
    expected = [0, 0, 0, 1, 7, 37, 197, 1172]
    g_family = [self.K(n) for n in range(8)]
    self.check_cycle_enumeration_integer_sequence(g_family, expected)