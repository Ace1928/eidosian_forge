import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_union_all_attributes():
    g = nx.Graph()
    g.add_node(0, x=4)
    g.add_node(1, x=5)
    g.add_edge(0, 1, size=5)
    g.graph['name'] = 'g'
    h = g.copy()
    h.graph['name'] = 'h'
    h.graph['attr'] = 'attr'
    h.nodes[0]['x'] = 7
    j = g.copy()
    j.graph['name'] = 'j'
    j.graph['attr'] = 'attr'
    j.nodes[0]['x'] = 7
    ghj = nx.union_all([g, h, j], rename=('g', 'h', 'j'))
    assert set(ghj.nodes()) == {'h0', 'h1', 'g0', 'g1', 'j0', 'j1'}
    for n in ghj:
        graph, node = n
        assert ghj.nodes[n] == eval(graph).nodes[int(node)]
    assert ghj.graph['attr'] == 'attr'
    assert ghj.graph['name'] == 'j'