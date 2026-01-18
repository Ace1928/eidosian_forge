import random
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.algorithms.link_analysis.pagerank_alg import (
@pytest.mark.parametrize('alg', (nx.pagerank, _pagerank_python))
def test_pagerank(self, alg):
    G = self.G
    p = alg(G, alpha=0.9, tol=1e-08)
    for n in G:
        assert p[n] == pytest.approx(G.pagerank[n], abs=0.0001)
    nstart = {n: random.random() for n in G}
    p = alg(G, alpha=0.9, tol=1e-08, nstart=nstart)
    for n in G:
        assert p[n] == pytest.approx(G.pagerank[n], abs=0.0001)