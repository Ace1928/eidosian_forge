import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_shortest_augmenting_path_two_phase():
    k = 5
    p = 1000
    G = nx.DiGraph()
    for i in range(k):
        G.add_edge('s', (i, 0), capacity=1)
        nx.add_path(G, ((i, j) for j in range(p)), capacity=1)
        G.add_edge((i, p - 1), 't', capacity=1)
    R = shortest_augmenting_path(G, 's', 't', two_phase=True)
    assert R.graph['flow_value'] == k
    R = shortest_augmenting_path(G, 's', 't', two_phase=False)
    assert R.graph['flow_value'] == k