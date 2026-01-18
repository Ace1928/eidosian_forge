from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_lca_routine_bails_on_DAGs(self):
    G = nx.DiGraph([(3, 4), (5, 4)])
    pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))