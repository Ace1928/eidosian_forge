import io
import time
import pytest
import networkx as nx
def test_slice_and_spell(self):
    G = nx.Graph()
    G.add_node(0, label='1', color='green')
    G.nodes[0]['spells'] = [(1, 2)]
    fh = io.BytesIO()
    nx.write_gexf(G, fh)
    fh.seek(0)
    H = nx.read_gexf(fh, node_type=int)
    assert sorted(G.nodes()) == sorted(H.nodes())
    assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))
    G = nx.Graph()
    G.add_node(0, label='1', color='green')
    G.nodes[0]['slices'] = [(1, 2)]
    fh = io.BytesIO()
    nx.write_gexf(G, fh, version='1.1draft')
    fh.seek(0)
    H = nx.read_gexf(fh, node_type=int)
    assert sorted(G.nodes()) == sorted(H.nodes())
    assert sorted((sorted(e) for e in G.edges())) == sorted((sorted(e) for e in H.edges()))