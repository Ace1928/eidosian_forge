from math import sqrt
import pytest
import networkx as nx
@pytest.mark.parametrize('method', methods)
def test_two_nodes_multigraph(self, method):
    pytest.importorskip('scipy')
    G = nx.MultiGraph()
    G.add_edge(0, 0, spam=100000000.0)
    G.add_edge(0, 1, spam=1)
    G.add_edge(0, 1, spam=-2)
    A = -3 * nx.laplacian_matrix(G, weight='spam')
    assert nx.algebraic_connectivity(G, weight='spam', tol=1e-12, method=method) == pytest.approx(6, abs=1e-07)
    x = nx.fiedler_vector(G, weight='spam', tol=1e-12, method=method)
    check_eigenvector(A, 6, x)