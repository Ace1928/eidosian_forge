import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_shown_node(self):
    induced_subgraph = nx.filters.show_nodes([2, 3, 111])
    gview = self.gview
    G = gview(self.G, filter_node=induced_subgraph)
    assert set(G.nodes) == {2, 3}
    if G.is_directed():
        assert list(G[3]) == []
    else:
        assert list(G[3]) == [2]
    assert list(G[2]) == [3]
    pytest.raises(KeyError, G.__getitem__, 4)
    pytest.raises(KeyError, G.__getitem__, 112)
    pytest.raises(KeyError, G.__getitem__, 111)
    assert G.degree(3) == (3 if G.is_multigraph() else 1)
    assert G.size() == (3 if G.is_multigraph() else 1)