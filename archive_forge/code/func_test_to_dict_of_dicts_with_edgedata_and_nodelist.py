import pytest
import networkx as nx
from networkx.convert import (
from networkx.generators.classic import barbell_graph, cycle_graph
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_to_dict_of_dicts_with_edgedata_and_nodelist():
    G = nx.path_graph(5)
    nodelist = [2, 3, 4]
    expected = {2: {3: 10}, 3: {2: 10, 4: 10}, 4: {3: 10}}
    assert nx.to_dict_of_dicts(G, nodelist=nodelist, edge_data=10) == expected