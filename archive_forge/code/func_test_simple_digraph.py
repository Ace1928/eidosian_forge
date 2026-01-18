import bz2
import importlib.resources
import os
import pickle
import pytest
import networkx as nx
def test_simple_digraph(self):
    G = nx.DiGraph()
    G.add_node('a', demand=-5)
    G.add_node('d', demand=5)
    G.add_edge('a', 'b', weight=3, capacity=4)
    G.add_edge('a', 'c', weight=6, capacity=10)
    G.add_edge('b', 'd', weight=1, capacity=9)
    G.add_edge('c', 'd', weight=2, capacity=5)
    flowCost, H = nx.network_simplex(G)
    soln = {'a': {'b': 4, 'c': 1}, 'b': {'d': 4}, 'c': {'d': 1}, 'd': {}}
    assert flowCost == 24
    assert nx.min_cost_flow_cost(G) == 24
    assert H == soln
    assert nx.min_cost_flow(G) == soln
    assert nx.cost_of_flow(G, H) == 24
    flowCost, H = nx.capacity_scaling(G)
    assert flowCost == 24
    assert nx.cost_of_flow(G, H) == 24
    assert H == soln