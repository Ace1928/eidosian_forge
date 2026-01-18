from itertools import chain, islice, tee
from math import inf
from random import shuffle
import pytest
import networkx as nx
from networkx.algorithms.traversal.edgedfs import FORWARD, REVERSE
def test_simple_cycles_bound_corner_cases(self):
    G = nx.cycle_graph(4)
    DG = nx.cycle_graph(4, create_using=nx.DiGraph)
    assert list(nx.simple_cycles(G, length_bound=0)) == []
    assert list(nx.simple_cycles(DG, length_bound=0)) == []
    assert list(nx.chordless_cycles(G, length_bound=0)) == []
    assert list(nx.chordless_cycles(DG, length_bound=0)) == []