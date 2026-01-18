import pytest
import networkx as nx
def test_wrong_nodes_prev_cc_raises(self):
    with pytest.raises(nx.NetworkXError):
        G = self.undirected_G.copy()
        edge = self.pick_add_edge(G)
        insert = True
        prev_cc = self.undirected_G_cc.copy()
        num_nodes = len(prev_cc)
        prev_cc.pop(0)
        prev_cc[num_nodes] = 0.5
        nx.incremental_closeness_centrality(G, edge, prev_cc, insert)