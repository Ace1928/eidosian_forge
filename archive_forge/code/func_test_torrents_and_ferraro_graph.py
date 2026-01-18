import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity.kcutsets import _is_separating_set
@pytest.mark.slow
def test_torrents_and_ferraro_graph():
    G = torrents_and_ferraro_graph()
    _check_separating_sets(G)