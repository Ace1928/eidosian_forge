import pytest
import networkx as nx
def test_nospan(self):
    expected = {(3, 4), (4, 3)}
    assert next(nx.local_bridges(self.BB, with_span=False)) in expected
    assert set(nx.local_bridges(self.square, with_span=False)) == self.square.edges
    assert list(nx.local_bridges(self.tri, with_span=False)) == []