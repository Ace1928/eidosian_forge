import pytest
import networkx as nx
from networkx.convert import (
from networkx.generators.classic import barbell_graph, cycle_graph
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_to_dict_of_dicts_with_edgedata_multigraph():
    """Multi edge data overwritten when edge_data != None"""
    G = nx.MultiGraph()
    G.add_edge(0, 1, key='a')
    G.add_edge(0, 1, key='b')
    expected = {0: {1: 10}, 1: {0: 10}}
    assert nx.to_dict_of_dicts(G, edge_data=10) == expected