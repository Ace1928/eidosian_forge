from itertools import chain, combinations, product
import pytest
import networkx as nx
def test_tree_all_pairs_lca_subtrees(self):
    ans = dict(tree_all_pairs_lca(self.DG, 1))
    gold = {pair: lca for pair, lca in self.gold.items() if all((n in (1, 3, 4) for n in pair))}
    self.assert_has_same_pairs(gold, ans)