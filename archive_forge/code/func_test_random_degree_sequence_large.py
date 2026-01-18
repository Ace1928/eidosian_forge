import pytest
import networkx as nx
def test_random_degree_sequence_large():
    G1 = nx.fast_gnp_random_graph(100, 0.1, seed=42)
    d1 = (d for n, d in G1.degree())
    G2 = nx.random_degree_sequence_graph(d1, seed=42)
    d2 = (d for n, d in G2.degree())
    assert sorted(d1) == sorted(d2)