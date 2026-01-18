import random
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_subgraph(self):
    assert self.G.subgraph([0, 1, 2, 4]).adj == nx.subgraph(self.G, [0, 1, 2, 4]).adj
    assert self.DG.subgraph([0, 1, 2, 4]).adj == nx.subgraph(self.DG, [0, 1, 2, 4]).adj
    assert self.G.subgraph([0, 1, 2, 4]).adj == nx.induced_subgraph(self.G, [0, 1, 2, 4]).adj
    assert self.DG.subgraph([0, 1, 2, 4]).adj == nx.induced_subgraph(self.DG, [0, 1, 2, 4]).adj
    H = nx.induced_subgraph(self.G.subgraph([0, 1, 2, 4]), [0, 1, 4])
    assert H._graph is not self.G
    assert H.adj == self.G.subgraph([0, 1, 4]).adj