import numbers
import pytest
import networkx as nx
from ..generators import (
def test_preferential_attachment(self):
    aseq = [3, 2, 1, 1]
    G = preferential_attachment_graph(aseq, 0.5)
    assert G.is_multigraph()
    assert not G.is_directed()
    G = preferential_attachment_graph(aseq, 0.5, create_using=nx.Graph)
    assert not G.is_multigraph()
    assert not G.is_directed()
    pytest.raises(nx.NetworkXError, preferential_attachment_graph, aseq, 0.5, create_using=nx.DiGraph())
    pytest.raises(nx.NetworkXError, preferential_attachment_graph, aseq, 0.5, create_using=nx.DiGraph())
    pytest.raises(nx.NetworkXError, preferential_attachment_graph, aseq, 0.5, create_using=nx.DiGraph())