import math
import pytest
import networkx as nx
class TestKatzCentralityDirectedNumpy(TestKatzCentralityDirected):

    @classmethod
    def setup_class(cls):
        global np
        np = pytest.importorskip('numpy')
        pytest.importorskip('scipy')
        super().setup_class()

    def test_katz_centrality_weighted(self):
        G = self.G
        alpha = self.G.alpha
        p = nx.katz_centrality_numpy(G, alpha, weight='weight')
        for a, b in zip(list(p.values()), self.G.evc):
            assert a == pytest.approx(b, abs=1e-07)

    def test_katz_centrality_unweighted(self):
        H = self.H
        alpha = self.H.alpha
        p = nx.katz_centrality_numpy(H, alpha, weight='weight')
        for a, b in zip(list(p.values()), self.H.evc):
            assert a == pytest.approx(b, abs=1e-07)