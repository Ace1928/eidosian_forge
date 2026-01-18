from itertools import combinations
import pytest
import networkx as nx
def test_path_graph_dsep(path_graph):
    """Example-based test of d-separation for path_graph."""
    assert nx.d_separated(path_graph, {0}, {2}, {1})
    assert not nx.d_separated(path_graph, {0}, {2}, {})