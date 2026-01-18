from itertools import chain, combinations
import pytest
import networkx as nx
def test_weight_argument(self):
    G = nx.MultiDiGraph()
    G.add_edge(1, 2, weight=1.41)
    G.add_edge(2, 1, weight=1.41)
    G.add_edge(2, 3)
    G.add_edge(3, 4, weight=3.14)
    truth = {frozenset({1, 2}), frozenset({3, 4})}
    self._check_communities(G, truth, weight='weight')