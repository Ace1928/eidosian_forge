import random
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.algorithms.link_analysis.pagerank_alg import (
def test_numpy_pagerank(self):
    G = self.G
    p = _pagerank_numpy(G, alpha=0.9)
    for n in G:
        assert p[n] == pytest.approx(G.pagerank[n], abs=0.0001)