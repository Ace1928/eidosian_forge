import pytest
import networkx as nx
def test_percolation_example1b():
    """percolation centrality: example 1a"""
    G = example1b_G()
    p = nx.percolation_centrality(G)
    p_answer = {4: 0.825, 6: 0.4}
    for n, k in p_answer.items():
        assert p[n] == pytest.approx(k, abs=0.001)