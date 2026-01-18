import random
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.algorithms.link_analysis.pagerank_alg import (
def test_scipy_pagerank_max_iter(self):
    with pytest.raises(nx.PowerIterationFailedConvergence):
        _pagerank_scipy(self.G, max_iter=0)