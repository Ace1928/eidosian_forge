import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import minimum_st_edge_cut, minimum_st_node_cut
from networkx.utils import arbitrary_element
def test_petersen_cutset():
    G = nx.petersen_graph()
    for flow_func in flow_funcs:
        kwargs = {'flow_func': flow_func}
        errmsg = f'Assertion failed in function: {flow_func.__name__}'
        edge_cut = nx.minimum_edge_cut(G, **kwargs)
        assert 3 == len(edge_cut), errmsg
        H = G.copy()
        H.remove_edges_from(edge_cut)
        assert not nx.is_connected(H), errmsg
        node_cut = nx.minimum_node_cut(G, **kwargs)
        assert 3 == len(node_cut), errmsg
        H = G.copy()
        H.remove_nodes_from(node_cut)
        assert not nx.is_connected(H), errmsg