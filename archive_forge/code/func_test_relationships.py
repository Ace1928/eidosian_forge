from pytest import approx
from networkx import is_connected, neighbors
from networkx.generators.internet_as_graphs import random_internet_as_graph
def test_relationships(self):
    for i in self.T:
        assert len(self.providers[i]) == 0
    for i in self.C:
        assert len(self.customers[i]) == 0
    for i in self.CP:
        assert len(self.customers[i]) == 0
    for i in self.G.nodes():
        assert len(self.customers[i].intersection(self.providers[i])) == 0
    for i, j in self.G.edges():
        if self.G.edges[i, j]['type'] == 'peer':
            assert j not in self.customers[i]
            assert i not in self.customers[j]
            assert j not in self.providers[i]
            assert i not in self.providers[j]