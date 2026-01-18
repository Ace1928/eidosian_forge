import pytest
import networkx as nx
def test_zero_centrality(self):
    G = nx.path_graph(3)
    prev_cc = nx.closeness_centrality(G)
    edge = self.pick_remove_edge(G)
    test_cc = nx.incremental_closeness_centrality(G, edge, prev_cc, insertion=False)
    G.remove_edges_from([edge])
    real_cc = nx.closeness_centrality(G)
    shared_items = set(test_cc.items()) & set(real_cc.items())
    assert len(shared_items) == len(real_cc)
    assert 0 in test_cc.values()