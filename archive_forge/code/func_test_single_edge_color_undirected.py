import itertools
import os
import warnings
import pytest
import networkx as nx
@pytest.mark.parametrize(('edge_color', 'expected'), ((None, 'black'), ('r', 'red'), (['r'], 'red'), ((1.0, 1.0, 0.0), 'yellow'), ([(1.0, 1.0, 0.0)], 'yellow'), ((0, 1, 0, 1), 'lime'), ([(0, 1, 0, 1)], 'lime'), ('#0000ff', 'blue'), (['#0000ff'], 'blue')))
@pytest.mark.parametrize('edgelist', (None, [(0, 1)]))
def test_single_edge_color_undirected(edge_color, expected, edgelist):
    """Tests ways of specifying all edges have a single color for edges
    drawn with a LineCollection"""
    G = nx.path_graph(3)
    drawn_edges = nx.draw_networkx_edges(G, pos=nx.random_layout(G), edgelist=edgelist, edge_color=edge_color)
    assert mpl.colors.same_color(drawn_edges.get_color(), expected)