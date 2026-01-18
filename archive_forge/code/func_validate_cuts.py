import pytest
import networkx as nx
from networkx.algorithms.flow import (
def validate_cuts(G, s, t, solnValue, partition, capacity, flow_func):
    errmsg = f'Assertion failed in function: {flow_func.__name__}'
    assert all((n in G for n in partition[0])), errmsg
    assert all((n in G for n in partition[1])), errmsg
    cutset = compute_cutset(G, partition)
    assert all((G.has_edge(u, v) for u, v in cutset)), errmsg
    assert solnValue == sum((G[u][v][capacity] for u, v in cutset)), errmsg
    H = G.copy()
    H.remove_edges_from(cutset)
    if not G.is_directed():
        assert not nx.is_connected(H), errmsg
    else:
        assert not nx.is_strongly_connected(H), errmsg