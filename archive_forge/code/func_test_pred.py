import pytest
import networkx as nx
from networkx.utils import edges_equal
def test_pred(self):
    edges_gone = self.hide_edges_filter(self.hide_edges)
    hide_nodes = [4, 5, 111]
    nodes_gone = nx.filters.hide_nodes(hide_nodes)
    G = self.gview(self.G, filter_node=nodes_gone, filter_edge=edges_gone)
    assert list(G.pred[2]) == [1]
    assert list(G.pred[6]) == []