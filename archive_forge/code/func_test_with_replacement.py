import pytest
import networkx as nx
from networkx.classes import Graph, MultiDiGraph
from networkx.generators.directed import (
def test_with_replacement(self):
    n = 10
    k = 3
    G = random_uniform_k_out_graph(n, k, with_replacement=True)
    assert G.is_multigraph()
    assert all((d == k for v, d in G.out_degree()))
    n = 10
    k = 9
    G = random_uniform_k_out_graph(n, k, with_replacement=False, self_loops=False)
    assert nx.number_of_selfloops(G) == 0
    assert all((d == k for v, d in G.out_degree()))