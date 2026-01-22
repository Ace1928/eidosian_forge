from itertools import chain, combinations, product
import pytest
import networkx as nx
class TestTreeLCA:

    @classmethod
    def setup_class(cls):
        cls.DG = nx.DiGraph()
        edges = [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (2, 6)]
        cls.DG.add_edges_from(edges)
        cls.ans = dict(tree_all_pairs_lca(cls.DG, 0))
        gold = {(n, n): n for n in cls.DG}
        gold.update({(0, i): 0 for i in range(1, 7)})
        gold.update({(1, 2): 0, (1, 3): 1, (1, 4): 1, (1, 5): 0, (1, 6): 0, (2, 3): 0, (2, 4): 0, (2, 5): 2, (2, 6): 2, (3, 4): 1, (3, 5): 0, (3, 6): 0, (4, 5): 0, (4, 6): 0, (5, 6): 2})
        cls.gold = gold

    @staticmethod
    def assert_has_same_pairs(d1, d2):
        for a, b in ((min(pair), max(pair)) for pair in chain(d1, d2)):
            assert get_pair(d1, a, b) == get_pair(d2, a, b)

    def test_tree_all_pairs_lca_default_root(self):
        assert dict(tree_all_pairs_lca(self.DG)) == self.ans

    def test_tree_all_pairs_lca_return_subset(self):
        test_pairs = [(0, 1), (0, 1), (1, 0)]
        ans = dict(tree_all_pairs_lca(self.DG, 0, test_pairs))
        assert (0, 1) in ans and (1, 0) in ans
        assert len(ans) == 2

    def test_tree_all_pairs_lca(self):
        all_pairs = chain(combinations(self.DG, 2), ((node, node) for node in self.DG))
        ans = dict(tree_all_pairs_lca(self.DG, 0, all_pairs))
        self.assert_has_same_pairs(ans, self.ans)

    def test_tree_all_pairs_gold_example(self):
        ans = dict(tree_all_pairs_lca(self.DG))
        self.assert_has_same_pairs(self.gold, ans)

    def test_tree_all_pairs_lca_invalid_input(self):
        empty_digraph = tree_all_pairs_lca(nx.DiGraph())
        pytest.raises(nx.NetworkXPointlessConcept, list, empty_digraph)
        bad_pairs_digraph = tree_all_pairs_lca(self.DG, pairs=[(-1, -2)])
        pytest.raises(nx.NodeNotFound, list, bad_pairs_digraph)

    def test_tree_all_pairs_lca_subtrees(self):
        ans = dict(tree_all_pairs_lca(self.DG, 1))
        gold = {pair: lca for pair, lca in self.gold.items() if all((n in (1, 3, 4) for n in pair))}
        self.assert_has_same_pairs(gold, ans)

    def test_tree_all_pairs_lca_disconnected_nodes(self):
        G = nx.DiGraph()
        G.add_node(1)
        assert {(1, 1): 1} == dict(tree_all_pairs_lca(G))
        G.add_node(0)
        assert {(1, 1): 1} == dict(tree_all_pairs_lca(G, 1))
        assert {(0, 0): 0} == dict(tree_all_pairs_lca(G, 0))
        pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))

    def test_tree_all_pairs_lca_error_if_input_not_tree(self):
        G = nx.DiGraph([(1, 2), (2, 1)])
        pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))
        G = nx.DiGraph([(0, 2), (1, 2)])
        pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))

    def test_tree_all_pairs_lca_generator(self):
        pairs = iter([(0, 1), (0, 1), (1, 0)])
        some_pairs = dict(tree_all_pairs_lca(self.DG, 0, pairs))
        assert (0, 1) in some_pairs and (1, 0) in some_pairs
        assert len(some_pairs) == 2

    def test_tree_all_pairs_lca_nonexisting_pairs_exception(self):
        lca = tree_all_pairs_lca(self.DG, 0, [(-1, -1)])
        pytest.raises(nx.NodeNotFound, list, lca)
        lca = tree_all_pairs_lca(self.DG, None, [(-1, -1)])
        pytest.raises(nx.NodeNotFound, list, lca)

    def test_tree_all_pairs_lca_routine_bails_on_DAGs(self):
        G = nx.DiGraph([(3, 4), (5, 4)])
        pytest.raises(nx.NetworkXError, list, tree_all_pairs_lca(G))

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

    def test_tree_all_pairs_lca_trees_without_LCAs(self):
        G = nx.DiGraph()
        G.add_node(3)
        ans = list(tree_all_pairs_lca(G))
        assert ans == [((3, 3), 3)]