import pytest
import networkx as nx
from networkx.algorithms.similarity import (
from networkx.generators.classic import (
def test_simrank_between_versions(self):
    G = nx.cycle_graph(5)
    expected_python_tol4 = {0: 1, 1: 0.394512499239852, 2: 0.5703550452791322, 3: 0.5703550452791323, 4: 0.394512499239852}
    expected_numpy_tol4 = {0: 1.0, 1: 0.3947180735764555, 2: 0.570482097206368, 3: 0.570482097206368, 4: 0.3947180735764555}
    actual = nx.simrank_similarity(G, source=0)
    assert expected_numpy_tol4 == pytest.approx(actual, abs=1e-07)
    assert expected_python_tol4 != pytest.approx(actual, abs=0.0001)
    assert expected_python_tol4 == pytest.approx(actual, abs=0.001)
    actual = nx.similarity._simrank_similarity_python(G, source=0)
    assert expected_python_tol4 == pytest.approx(actual, abs=1e-07)
    assert expected_numpy_tol4 != pytest.approx(actual, abs=0.0001)
    assert expected_numpy_tol4 == pytest.approx(actual, abs=0.001)