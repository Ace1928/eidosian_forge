from itertools import chain, combinations
import pytest
import networkx as nx
def test_several_communities(self):
    ground_truth = {frozenset(range(3 * i, 3 * (i + 1))) for i in range(5)}
    edges = chain.from_iterable((combinations(c, 2) for c in ground_truth))
    G = nx.Graph(edges)
    self._check_communities(G, ground_truth)