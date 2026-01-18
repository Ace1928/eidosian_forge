import os
import tempfile
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
def test_round_trip_empty_graph(self):
    G = nx.Graph()
    A = nx.nx_agraph.to_agraph(G)
    H = nx.nx_agraph.from_agraph(A)
    AA = nx.nx_agraph.to_agraph(H)
    HH = nx.nx_agraph.from_agraph(AA)
    assert graphs_equal(H, HH)
    G.graph['graph'] = {}
    G.graph['node'] = {}
    G.graph['edge'] = {}
    assert graphs_equal(G, HH)