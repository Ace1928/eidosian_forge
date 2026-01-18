from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_simple_cycles_acyclic_tournament(self):
    n = 10
    G = nx.DiGraph(((x, y) for x in range(n) for y in range(x)))
    self.check_cycle_algorithm(G, [])
    self.check_cycle_algorithm(G, [], chordless=True)
    for k in range(n + 1):
        self.check_cycle_algorithm(G, [], length_bound=k)
        self.check_cycle_algorithm(G, [], length_bound=k, chordless=True)