from itertools import chain, combinations, product
import pytest
import networkx as nx
class TestMultiTreeLCA(TestTreeLCA):

    @classmethod
    def setup_class(cls):
        cls.DG = nx.MultiDiGraph()
        edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        cls.DG.add_edges_from(edges)
        cls.ans = dict(tree_all_pairs_lca(cls.DG, 0))
        cls.DG.add_edges_from(edges)
        gold = {(n, n): n for n in cls.DG}
        gold.update({(0, i): 0 for i in range(1, 7)})
        gold.update({(1, 2): 0, (1, 3): 1, (1, 4): 1, (1, 5): 0, (1, 6): 0, (2, 3): 0, (2, 4): 0, (2, 5): 2, (2, 6): 2, (3, 4): 1, (3, 5): 0, (3, 6): 0, (4, 5): 0, (4, 6): 0, (5, 6): 2})
        cls.gold = gold