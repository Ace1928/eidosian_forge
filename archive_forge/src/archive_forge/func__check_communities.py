from itertools import chain, combinations
import pytest
import networkx as nx
def _check_communities(self, G, truth, weight=None, seed=None):
    C = nx.community.fast_label_propagation_communities(G, weight=weight, seed=seed)
    assert {frozenset(c) for c in C} == truth