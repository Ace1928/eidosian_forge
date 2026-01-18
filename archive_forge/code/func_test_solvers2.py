import pytest
import networkx as nx
from networkx import approximate_current_flow_betweenness_centrality as approximate_cfbc
from networkx import edge_current_flow_betweenness_centrality as edge_current_flow
def test_solvers2(self):
    """Betweenness centrality: alternate solvers"""
    G = nx.complete_graph(4)
    for solver in ['full', 'lu', 'cg']:
        b = nx.current_flow_betweenness_centrality(G, normalized=False, solver=solver)
        b_answer = {0: 0.75, 1: 0.75, 2: 0.75, 3: 0.75}
        for n in sorted(G):
            assert b[n] == pytest.approx(b_answer[n], abs=1e-07)