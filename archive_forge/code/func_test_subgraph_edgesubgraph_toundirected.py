import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
def test_subgraph_edgesubgraph_toundirected(self):
    G = self.G.copy()
    SG = G.subgraph([4, 5, 6])
    SSG = SG.edge_subgraph([(4, 5), (5, 4)])
    USSG = SSG.to_undirected()
    assert list(USSG) == [4, 5]
    assert sorted(USSG.edges) == [(4, 5)]