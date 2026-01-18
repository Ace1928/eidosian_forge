import pytest
import networkx as nx
def test_trophic_levels_singular_with_basal():
    """Should fail to compute if there are any parts of the graph which are not
    reachable from any basal node (with in-degree zero).
    """
    G = nx.DiGraph()
    G.add_edge('a', 'b')
    G.add_edge('c', 'b')
    G.add_edge('d', 'b')
    G.add_edge('c', 'd')
    G.add_edge('d', 'c')
    with pytest.raises(nx.NetworkXError) as e:
        nx.trophic_levels(G)
    msg = 'Trophic levels are only defined for graphs where every node ' + 'has a path from a basal node (basal nodes are nodes with no ' + 'incoming edges).'
    assert msg in str(e.value)
    G = nx.DiGraph()
    G.add_edge('a', 'b')
    G.add_edge('c', 'b')
    G.add_edge('c', 'c')
    with pytest.raises(nx.NetworkXError) as e:
        nx.trophic_levels(G)
    msg = 'Trophic levels are only defined for graphs where every node ' + 'has a path from a basal node (basal nodes are nodes with no ' + 'incoming edges).'
    assert msg in str(e.value)