import itertools
from collections import defaultdict
from random import sample
import pytest
import networkx as nx
@pytest.mark.parametrize('N', [5, 10])
def test_triadic_census_on_random_graph(N):
    G = nx.binomial_graph(N, 0.3, directed=True, seed=42)
    tc1 = nx.triadic_census(G)
    tbt = nx.triads_by_type(G)
    tc2 = {tt: len(tbt[tt]) for tt in tc1}
    assert tc1 == tc2
    for n in G:
        tc1 = nx.triadic_census(G, nodelist={n})
        tc2 = {tt: sum((1 for t in tbt.get(tt, []) if n in t)) for tt in tc1}
        assert tc1 == tc2
    for ns in itertools.combinations(G, 2):
        ns = set(ns)
        tc1 = nx.triadic_census(G, nodelist=ns)
        tc2 = {tt: sum((1 for t in tbt.get(tt, []) if any((n in ns for n in t)))) for tt in tc1}
        assert tc1 == tc2
    for ns in itertools.combinations(G, 3):
        ns = set(ns)
        tc1 = nx.triadic_census(G, nodelist=ns)
        tc2 = {tt: sum((1 for t in tbt.get(tt, []) if any((n in ns for n in t)))) for tt in tc1}
        assert tc1 == tc2