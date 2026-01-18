import pytest
import networkx as nx
def test_trophic_levels_even_more_complex():
    matrix = np.array([[0, 0, 0, 0, 0], [0, 1, 0, 1, 0], [1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0]])
    K = np.array([[1, 0, -1, 0, 0], [0, 0.5, 0, -0.5, 0], [0, 0, 1, 0, 0], [0, -0.5, 0, 1, -0.5], [0, 0, 0, 0, 1]])
    result_1 = np.ravel(np.linalg.inv(K) @ np.ones(5))
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    result_2 = nx.trophic_levels(G)
    for ind in range(5):
        assert result_1[ind] == pytest.approx(result_2[ind], abs=1e-07)