import pytest
import networkx
import networkx as nx
from .historical_tests import HistoricalTests
def test_reverse2(self):
    H = nx.DiGraph()
    foo = [H.add_edge(u, u + 1) for u in range(5)]
    HR = H.reverse()
    for u in range(5):
        assert HR.has_edge(u + 1, u)