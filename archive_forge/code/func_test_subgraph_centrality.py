import pytest
import networkx as nx
from networkx.algorithms.centrality.subgraph_alg import (
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