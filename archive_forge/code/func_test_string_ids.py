import json
import pytest
import networkx as nx
from networkx.readwrite.json_graph import node_link_data, node_link_graph
def test_string_ids(self):
    q = 'qualité'
    G = nx.DiGraph()
    G.add_node('A')
    G.add_node(q)
    G.add_edge('A', q)
    data = node_link_data(G)
    assert data['links'][0]['source'] == 'A'
    assert data['links'][0]['target'] == q
    H = node_link_graph(data)
    assert nx.is_isomorphic(G, H)