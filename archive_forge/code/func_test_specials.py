import io
import time
import pytest
import networkx as nx
def test_specials(self):
    from math import isnan
    inf, nan = (float('inf'), float('nan'))
    G = nx.Graph()
    G.add_node(1, testattr=inf, strdata='inf', key='a')
    G.add_node(2, testattr=nan, strdata='nan', key='b')
    G.add_node(3, testattr=-inf, strdata='-inf', key='c')
    fh = io.BytesIO()
    nx.write_gexf(G, fh)
    fh.seek(0)
    filetext = fh.read()
    fh.seek(0)
    H = nx.read_gexf(fh, node_type=int)
    assert b'INF' in filetext
    assert b'NaN' in filetext
    assert b'-INF' in filetext
    assert H.nodes[1]['testattr'] == inf
    assert isnan(H.nodes[2]['testattr'])
    assert H.nodes[3]['testattr'] == -inf
    assert H.nodes[1]['strdata'] == 'inf'
    assert H.nodes[2]['strdata'] == 'nan'
    assert H.nodes[3]['strdata'] == '-inf'
    assert H.nodes[1]['networkx_key'] == 'a'
    assert H.nodes[2]['networkx_key'] == 'b'
    assert H.nodes[3]['networkx_key'] == 'c'