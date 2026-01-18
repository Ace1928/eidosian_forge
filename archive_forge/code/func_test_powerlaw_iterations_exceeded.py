import pytest
import networkx as nx
def test_powerlaw_iterations_exceeded():
    with pytest.raises(nx.ExceededMaxIterations, match='Could not create power law sequence'):
        n = 100
        tau1 = 2
        tau2 = 2
        mu = 1
        nx.LFR_benchmark_graph(n, tau1, tau2, mu, min_degree=2, max_iters=0)