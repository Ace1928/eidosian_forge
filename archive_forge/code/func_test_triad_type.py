import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_triad_type():
    """Tests the triad_type function."""
    G = nx.DiGraph({0: [], 1: [], 2: []})
    assert nx.triad_type(G) == '003'
    G = nx.DiGraph({0: [1], 1: [], 2: []})
    assert nx.triad_type(G) == '012'
    G = nx.DiGraph([(0, 1), (0, 2)])
    assert nx.triad_type(G) == '021D'
    G = nx.DiGraph({0: [1], 1: [0], 2: []})
    assert nx.triad_type(G) == '102'
    G = nx.DiGraph([(0, 1), (2, 1)])
    assert nx.triad_type(G) == '021U'
    G = nx.DiGraph([(0, 1), (1, 2)])
    assert nx.triad_type(G) == '021C'
    G = nx.DiGraph([(0, 1), (1, 0), (2, 1)])
    assert nx.triad_type(G) == '111D'
    G = nx.DiGraph([(0, 1), (1, 0), (1, 2)])
    assert nx.triad_type(G) == '111U'
    G = nx.DiGraph([(0, 1), (1, 2), (0, 2)])
    assert nx.triad_type(G) == '030T'
    G = nx.DiGraph([(0, 1), (1, 2), (2, 0)])
    assert nx.triad_type(G) == '030C'
    G = nx.DiGraph([(0, 1), (1, 0), (2, 0), (0, 2)])
    assert nx.triad_type(G) == '201'
    G = nx.DiGraph([(0, 1), (1, 0), (2, 0), (2, 1)])
    assert nx.triad_type(G) == '120D'
    G = nx.DiGraph([(0, 1), (1, 0), (0, 2), (1, 2)])
    assert nx.triad_type(G) == '120U'
    G = nx.DiGraph([(0, 1), (1, 0), (0, 2), (2, 1)])
    assert nx.triad_type(G) == '120C'
    G = nx.DiGraph([(0, 1), (1, 0), (2, 1), (1, 2), (0, 2)])
    assert nx.triad_type(G) == '210'
    G = nx.DiGraph([(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)])
    assert nx.triad_type(G) == '300'