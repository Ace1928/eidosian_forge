import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_white_harary_1():
    G = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))
    G.remove_node(7)
    for i in range(4, 7):
        G.add_edge(0, i)
    G = nx.disjoint_union(G, nx.complete_graph(4))
    G.remove_node(G.order() - 1)
    for i in range(7, 10):
        G.add_edge(0, i)
    for flow_func in flow_funcs:
        errmsg = f'Assertion failed in function: {flow_func.__name__}'
        assert 1 == nx.node_connectivity(G, flow_func=flow_func), errmsg
        assert 3 == nx.edge_connectivity(G, flow_func=flow_func), errmsg