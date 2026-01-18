from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_gold_example(self):
    ans = dict(tree_all_pairs_lca(self.DG))
    self.assert_has_same_pairs(self.gold, ans)