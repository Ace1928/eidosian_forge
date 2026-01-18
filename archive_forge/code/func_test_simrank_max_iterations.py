import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
@pytest.mark.parametrize('alg', simrank_algs)
def test_simrank_max_iterations(self, alg):
    G = nx.cycle_graph(5)
    pytest.raises(nx.ExceededMaxIterations, alg, G, max_iterations=10)