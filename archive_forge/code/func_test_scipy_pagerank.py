import random
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.algorithms.link_analysis.pagerank_alg import (
def test_scipy_pagerank(self):
    G = self.G
    p = _pagerank_scipy(G, alpha=0.9, tol=1e-08)
    for n in G:
        assert p[n] == pytest.approx(G.pagerank[n], abs=0.0001)
    personalize = {n: random.random() for n in G}
    p = _pagerank_scipy(G, alpha=0.9, tol=1e-08, personalization=personalize)
    nstart = {n: random.random() for n in G}
    p = _pagerank_scipy(G, alpha=0.9, tol=1e-08, nstart=nstart)
    for n in G:
        assert p[n] == pytest.approx(G.pagerank[n], abs=0.0001)