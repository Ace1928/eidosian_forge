from itertools import chain, combinations
import pytest
import networkx as nx
def test_simple_communities(self):
    G = nx.Graph(['ab', 'ac', 'bc', 'de', 'df', 'fe'])
    ground_truth = {frozenset('abc'), frozenset('def')}
    self._check_communities(G, ground_truth)