import pytest
from networkx import NetworkXError, cycle_graph
from networkx.algorithms.bipartite import complete_bipartite_graph, node_redundancy
def test_redundant_nodes():
    G = cycle_graph(6)
    edge = {0, 3}
    G.add_edge(*edge)
    redundancy = node_redundancy(G)
    for v in edge:
        assert redundancy[v] == 2 / 3
    for v in set(G) - edge:
        assert redundancy[v] == 1