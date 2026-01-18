from itertools import chain
import pytest
import networkx as nx
def test_single_node_graph(self):
    G = nx.DiGraph()
    G.add_node(0)
    assert nx.is_semiconnected(G)