import random
import pytest
import networkx as nx
from networkx.algorithms.approximation import maxcut
@pytest.mark.parametrize('f', (nx.approximation.randomized_partitioning, nx.approximation.one_exchange))
@pytest.mark.parametrize('graph_constructor', (nx.DiGraph, nx.MultiGraph))
def test_raises_on_directed_and_multigraphs(f, graph_constructor):
    G = graph_constructor([(0, 1), (1, 2)])
    with pytest.raises(nx.NetworkXNotImplemented):
        f(G)