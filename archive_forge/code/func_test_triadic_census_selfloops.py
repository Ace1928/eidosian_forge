import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
def test_triadic_census_selfloops():
    GG = nx.path_graph('abc', create_using=nx.DiGraph)
    expected = {'021C': 1}
    for n in GG:
        G = GG.copy()
        G.add_edge(n, n)
        tc = nx.triadic_census(G)
        assert expected == {typ: cnt for typ, cnt in tc.items() if cnt > 0}
    GG = nx.path_graph('abcde', create_using=nx.DiGraph)
    tbt = nx.triads_by_type(GG)
    for n in GG:
        GG.add_edge(n, n)
    tc = nx.triadic_census(GG)
    assert tc == {tt: len(tbt[tt]) for tt in tc}