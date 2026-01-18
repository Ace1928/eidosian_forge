from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_lca_trees_without_LCAs(self):
    G = nx.DiGraph()
    G.add_node(3)
    ans = list(tree_all_pairs_lca(G))
    assert ans == [((3, 3), 3)]