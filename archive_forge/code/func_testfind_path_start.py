import collections
import pytest
import networkx as nx
def testfind_path_start(self):
    find_path_start = nx.algorithms.euler._find_path_start
    G = nx.path_graph(6, create_using=nx.DiGraph)
    assert find_path_start(G) == 0
    edges = [(0, 1), (1, 2), (2, 0), (4, 0)]
    assert find_path_start(nx.DiGraph(edges)) == 4
    edges = [(0, 1), (1, 2), (2, 3), (2, 4)]
    assert find_path_start(nx.DiGraph(edges)) is None