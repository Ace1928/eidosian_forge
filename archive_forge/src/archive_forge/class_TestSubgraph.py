import pytest
import networkx as nx
from networkx.algorithms.centrality.subgraph_alg import (
class TestSubgraph:

    def test_subgraph_centrality(self):
        answer = {0: 1.5430806348152433, 1: 1.5430806348152433}
        result = subgraph_centrality(nx.path_graph(2))
        for k, v in result.items():
            assert answer[k] == pytest.approx(v, abs=1e-07)
        answer1 = {'1': 1.6445956054135658, 'Albert': 2.436825735871219, 'Aric': 2.4368257358712193, 'Dan': 3.130632849632817, 'Franck': 2.3876142275231915}
        G1 = nx.Graph([('Franck', 'Aric'), ('Aric', 'Dan'), ('Dan', 'Albert'), ('Albert', 'Franck'), ('Dan', '1'), ('Franck', 'Albert')])
        result1 = subgraph_centrality(G1)
        for k, v in result1.items():
            assert answer1[k] == pytest.approx(v, abs=1e-07)
        result1 = subgraph_centrality_exp(G1)
        for k, v in result1.items():
            assert answer1[k] == pytest.approx(v, abs=1e-07)

    def test_subgraph_centrality_big_graph(self):
        g199 = nx.complete_graph(199)
        g200 = nx.complete_graph(200)
        comm199 = nx.subgraph_centrality(g199)
        comm199_exp = nx.subgraph_centrality_exp(g199)
        comm200 = nx.subgraph_centrality(g200)
        comm200_exp = nx.subgraph_centrality_exp(g200)

    def test_communicability_betweenness_centrality_small(self):
        result = communicability_betweenness_centrality(nx.path_graph(2))
        assert result == {0: 0, 1: 0}
        result = communicability_betweenness_centrality(nx.path_graph(1))
        assert result == {0: 0}
        result = communicability_betweenness_centrality(nx.path_graph(0))
        assert result == {}
        answer = {0: 0.1411224421177313, 1: 1.0, 2: 0.1411224421177313}
        result = communicability_betweenness_centrality(nx.path_graph(3))
        for k, v in result.items():
            assert answer[k] == pytest.approx(v, abs=1e-07)
        result = communicability_betweenness_centrality(nx.complete_graph(3))
        for k, v in result.items():
            assert 0.49786143366223296 == pytest.approx(v, abs=1e-07)

    def test_communicability_betweenness_centrality(self):
        answer = {0: 0.07017447951484615, 1: 0.7156559870110799, 2: 0.7156559870110799, 3: 0.07017447951484615}
        result = communicability_betweenness_centrality(nx.path_graph(4))
        for k, v in result.items():
            assert answer[k] == pytest.approx(v, abs=1e-07)
        answer1 = {'1': 0.06003907419394952, 'Albert': 0.315470761661372, 'Aric': 0.3154707616613721, 'Dan': 0.682977786783162, 'Franck': 0.21977926617449497}
        G1 = nx.Graph([('Franck', 'Aric'), ('Aric', 'Dan'), ('Dan', 'Albert'), ('Albert', 'Franck'), ('Dan', '1'), ('Franck', 'Albert')])
        result1 = communicability_betweenness_centrality(G1)
        for k, v in result1.items():
            assert answer1[k] == pytest.approx(v, abs=1e-07)

    def test_estrada_index(self):
        answer = 1041.2470334195475
        result = estrada_index(nx.karate_club_graph())
        assert answer == pytest.approx(result, abs=1e-07)