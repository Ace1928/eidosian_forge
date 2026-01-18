import pytest
import networkx as nx
def test_trophic_levels_more_complex():
    matrix = np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]])
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    d = nx.trophic_levels(G)
    expected_result = [1, 2, 3, 4]
    for ind in range(4):
        assert d[ind] == pytest.approx(expected_result[ind], abs=1e-07)
    matrix = np.array([[0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 1], [0, 0, 0, 0]])
    G = nx.from_numpy_array(matrix, create_using=nx.DiGraph)
    d = nx.trophic_levels(G)
    expected_result = [1, 2, 2.5, 3.25]
    print('Calculated result: ', d)
    print('Expected Result: ', expected_result)
    for ind in range(4):
        assert d[ind] == pytest.approx(expected_result[ind], abs=1e-07)