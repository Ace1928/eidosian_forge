import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_simrank_numpy_no_source_no_target(self):
    G = nx.cycle_graph(5)
    expected = np.array([[1.0, 0.3947180735764555, 0.570482097206368, 0.570482097206368, 0.3947180735764555], [0.3947180735764555, 1.0, 0.3947180735764555, 0.570482097206368, 0.570482097206368], [0.570482097206368, 0.3947180735764555, 1.0, 0.3947180735764555, 0.570482097206368], [0.570482097206368, 0.570482097206368, 0.3947180735764555, 1.0, 0.3947180735764555], [0.3947180735764555, 0.570482097206368, 0.570482097206368, 0.3947180735764555, 1.0]])
    actual = nx.similarity._simrank_similarity_numpy(G)
    np.testing.assert_allclose(expected, actual, atol=1e-07)