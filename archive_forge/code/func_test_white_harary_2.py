import itertools
import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import (
def test_white_harary_2():
    G = nx.disjoint_union(nx.complete_graph(4), nx.complete_graph(4))
    G.add_edge(0, 4)
    assert 3 == min(nx.core_number(G).values())
    for flow_func in flow_funcs:
        errmsg = f'Assertion failed in function: {flow_func.__name__}'
        assert 1 == nx.node_connectivity(G, flow_func=flow_func), errmsg
        assert 1 == nx.edge_connectivity(G, flow_func=flow_func), errmsg