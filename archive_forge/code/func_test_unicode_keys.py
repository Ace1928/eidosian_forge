import json
import pytest
import networkx as nx
from networkx.readwrite.json_graph import node_link_data, node_link_graph
def test_unicode_keys(self):
    q = 'qualité'
    G = nx.Graph()
    G.add_node(1, **{q: q})
    s = node_link_data(G)
    output = json.dumps(s, ensure_ascii=False)
    data = json.loads(output)
    H = node_link_graph(data)
    assert H.nodes[1][q] == q