import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_parse_edgelist_with_data_list(example_graph):
    G = example_graph
    H = nx.parse_edgelist(['1 2 3', '2 3 27', '3 4 3.0'], nodetype=int, data=(('weight', float),))
    assert nodes_equal(G.nodes, H.nodes)
    assert edges_equal(G.edges(data=True), H.edges(data=True))