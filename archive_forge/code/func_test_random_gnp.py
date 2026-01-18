import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
def test_random_gnp():
    G = nx.gnp_random_graph(100, 0.1, seed=42)
    _check_separating_sets(G)