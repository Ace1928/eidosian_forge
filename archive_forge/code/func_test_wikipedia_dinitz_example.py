import pytest
import networkx as nx
from networkx.algorithms.flow import (
def test_wikipedia_dinitz_example(self):
    G = nx.DiGraph()
    G.add_edge('s', 1, capacity=10)
    G.add_edge('s', 2, capacity=10)
    G.add_edge(1, 3, capacity=4)
    G.add_edge(1, 4, capacity=8)
    G.add_edge(1, 2, capacity=2)
    G.add_edge(2, 4, capacity=9)
    G.add_edge(3, 't', capacity=10)
    G.add_edge(4, 3, capacity=6)
    G.add_edge(4, 't', capacity=10)
    solnFlows = {1: {2: 0, 3: 4, 4: 6}, 2: {4: 9}, 3: {'t': 9}, 4: {3: 5, 't': 10}, 's': {1: 10, 2: 9}, 't': {}}
    compare_flows_and_cuts(G, 's', 't', solnFlows, 19)