import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_triadic_census_four_path_nodelist():
    G = nx.path_graph('abcd', create_using=nx.DiGraph)
    expected_end = {'012': 2, '021C': 1}
    expected_mid = {'012': 1, '021C': 2}
    a_triad_census = nx.triadic_census(G, nodelist=['a'])
    assert expected_end == {typ: cnt for typ, cnt in a_triad_census.items() if cnt > 0}
    b_triad_census = nx.triadic_census(G, nodelist=['b'])
    assert expected_mid == {typ: cnt for typ, cnt in b_triad_census.items() if cnt > 0}
    c_triad_census = nx.triadic_census(G, nodelist=['c'])
    assert expected_mid == {typ: cnt for typ, cnt in c_triad_census.items() if cnt > 0}
    d_triad_census = nx.triadic_census(G, nodelist=['d'])
    assert expected_end == {typ: cnt for typ, cnt in d_triad_census.items() if cnt > 0}