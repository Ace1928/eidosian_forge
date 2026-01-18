import pytest
import networkx as nx
from networkx.algorithms import flow
from networkx.algorithms.connectivity import minimum_st_edge_cut, minimum_st_node_cut
from networkx.utils import arbitrary_element
def tests_min_cut_complete():
    G = nx.complete_graph(5)
    for interface_func in [nx.minimum_edge_cut, nx.minimum_node_cut]:
        for flow_func in flow_funcs:
            assert 4 == len(interface_func(G, flow_func=flow_func))