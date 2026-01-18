import random
import pytest
import networkx as nx
from networkx.classes.tests import dispatch_interface
from networkx.algorithms.link_analysis.pagerank_alg import (
@pytest.mark.parametrize('alg', (nx.pagerank, _pagerank_python))
def test_pagerank_max_iter(self, alg):
    with pytest.raises(nx.PowerIterationFailedConvergence):
        alg(self.G, max_iter=0)