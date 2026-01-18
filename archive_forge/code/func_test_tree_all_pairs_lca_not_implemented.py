from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_lca_not_implemented(self):
    NNI = nx.NetworkXNotImplemented
    G = nx.Graph([(0, 1)])
    with pytest.raises(NNI):
        next(tree_all_pairs_lca(G))
    with pytest.raises(NNI):
        next(all_pairs_lca(G))
    pytest.raises(NNI, nx.lowest_common_ancestor, G, 0, 1)
    G = nx.MultiGraph([(0, 1)])
    with pytest.raises(NNI):
        next(tree_all_pairs_lca(G))
    with pytest.raises(NNI):
        next(all_pairs_lca(G))
    pytest.raises(NNI, nx.lowest_common_ancestor, G, 0, 1)