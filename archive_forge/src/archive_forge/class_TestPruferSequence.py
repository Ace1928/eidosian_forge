from itertools import product
import pytest
import networkx as nx
from networkx.utils import edges_equal, nodes_equal
class TestPruferSequence:
    """Unit tests for the Prüfer sequence encoding and decoding
    functions.

    """

    def test_nontree(self):
        with pytest.raises(nx.NotATree):
            G = nx.cycle_graph(3)
            nx.to_prufer_sequence(G)

    def test_null_graph(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.to_prufer_sequence(nx.null_graph())

    def test_trivial_graph(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.to_prufer_sequence(nx.trivial_graph())

    def test_bad_integer_labels(self):
        with pytest.raises(KeyError):
            T = nx.Graph(nx.utils.pairwise('abc'))
            nx.to_prufer_sequence(T)

    def test_encoding(self):
        """Tests for encoding a tree as a Prüfer sequence using the
        iterative strategy.

        """
        tree = nx.Graph([(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)])
        sequence = nx.to_prufer_sequence(tree)
        assert sequence == [3, 3, 3, 4]

    def test_decoding(self):
        """Tests for decoding a tree from a Prüfer sequence."""
        sequence = [3, 3, 3, 4]
        tree = nx.from_prufer_sequence(sequence)
        assert nodes_equal(list(tree), list(range(6)))
        edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)]
        assert edges_equal(list(tree.edges()), edges)

    def test_decoding2(self):
        sequence = [2, 4, 0, 1, 3, 3]
        tree = nx.from_prufer_sequence(sequence)
        assert nodes_equal(list(tree), list(range(8)))
        edges = [(0, 1), (0, 4), (1, 3), (2, 4), (2, 5), (3, 6), (3, 7)]
        assert edges_equal(list(tree.edges()), edges)

    def test_inverse(self):
        """Tests that the encoding and decoding functions are inverses."""
        for T in nx.nonisomorphic_trees(4):
            T2 = nx.from_prufer_sequence(nx.to_prufer_sequence(T))
            assert nodes_equal(list(T), list(T2))
            assert edges_equal(list(T.edges()), list(T2.edges()))
        for seq in product(range(4), repeat=2):
            seq2 = nx.to_prufer_sequence(nx.from_prufer_sequence(seq))
            assert list(seq) == seq2