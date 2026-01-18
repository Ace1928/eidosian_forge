import pickle
from copy import deepcopy
import pytest
import networkx as nx
from networkx.classes import reportviews as rv
from networkx.classes.reportviews import NodeDataView
def test_xor(self):
    ev = self.eview(self.G)
    some_edges = {(0, 1, 0), (1, 0, 0), (0, 2, 0)}
    if self.G.is_directed():
        result = {(n, n + 1, 0) for n in range(1, 8)}
        result.update({(1, 0, 0), (0, 2, 0), (1, 2, 3)})
        assert ev ^ some_edges == result
        assert some_edges ^ ev == result
    else:
        result = {(n, n + 1, 0) for n in range(1, 8)}
        result.update({(0, 2, 0), (1, 2, 3)})
        assert ev ^ some_edges == result
        assert some_edges ^ ev == result