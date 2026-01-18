import warnings
import pytest
import networkx as nx
def test_smetric():
    g = nx.Graph()
    g.add_edge(1, 2)
    g.add_edge(2, 3)
    g.add_edge(2, 4)
    g.add_edge(1, 4)
    sm = nx.s_metric(g, normalized=False)
    assert sm == 19.0