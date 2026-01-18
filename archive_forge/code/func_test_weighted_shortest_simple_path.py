import random
import pytest
import networkx as nx
from networkx import convert_node_labels_to_integers as cnlti
from networkx.algorithms.simple_paths import (
from networkx.utils import arbitrary_element, pairwise
def test_weighted_shortest_simple_path():

    def cost_func(path):
        return sum((G.adj[u][v]['weight'] for u, v in zip(path, path[1:])))
    G = nx.complete_graph(5)
    weight = {(u, v): random.randint(1, 100) for u, v in G.edges()}
    nx.set_edge_attributes(G, weight, 'weight')
    cost = 0
    for path in nx.shortest_simple_paths(G, 0, 3, weight='weight'):
        this_cost = cost_func(path)
        assert cost <= this_cost
        cost = this_cost