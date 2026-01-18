import io
import os
import tempfile
import textwrap
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_parse_edgelist_no_data(example_graph):
    G = example_graph
    H = nx.parse_edgelist(['1 2', '2 3', '3 4'], nodetype=int)
    assert nodes_equal(G.nodes, H.nodes)
    assert edges_equal(G.edges, H.edges)