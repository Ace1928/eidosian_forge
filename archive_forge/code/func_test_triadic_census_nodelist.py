import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_triadic_census_nodelist():
    """Tests the triadic_census function."""
    G = nx.DiGraph()
    G.add_edges_from(['01', '02', '03', '04', '05', '12', '16', '51', '56', '65'])
    expected = {'030T': 2, '120C': 1, '210': 0, '120U': 0, '012': 9, '102': 3, '021U': 0, '111U': 0, '003': 8, '030C': 0, '021D': 9, '201': 0, '111D': 1, '300': 0, '120D': 0, '021C': 2}
    actual = {k: 0 for k in expected}
    for node in G.nodes():
        node_triad_census = nx.triadic_census(G, nodelist=[node])
        for triad_key in expected:
            actual[triad_key] += node_triad_census[triad_key]
    for k, v in actual.items():
        actual[k] //= 3
    assert expected == actual