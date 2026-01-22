import math
import pytest
import networkx as nx
class TestKatzEigenvectorVKatz:

    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip('numpy')
        pytest.importorskip('scipy')

    def test_eigenvector_v_katz_random(self):
        G = nx.gnp_random_graph(10, 0.5, seed=1234)
        l = max(np.linalg.eigvals(nx.adjacency_matrix(G).todense()))
        e = nx.eigenvector_centrality_numpy(G)
        k = nx.katz_centrality_numpy(G, 1.0 / l)
        for n in G:
            assert e[n] == pytest.approx(k[n], abs=1e-07)