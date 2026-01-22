import pytest
import networkx as nx
from networkx.algorithms import isomorphism as iso
class TestSubgraphIsomorphism:

    def test_isomorphism(self):
        g1 = nx.Graph()
        nx.add_cycle(g1, range(4))
        g2 = nx.Graph()
        nx.add_cycle(g2, range(4))
        g2.add_edges_from(list(zip(g2, range(4, 8))))
        ismags = iso.ISMAGS(g2, g1)
        assert list(ismags.subgraph_isomorphisms_iter(symmetry=True)) == [{n: n for n in g1.nodes}]

    def test_isomorphism2(self):
        g1 = nx.Graph()
        nx.add_path(g1, range(3))
        g2 = g1.copy()
        g2.add_edge(1, 3)
        ismags = iso.ISMAGS(g2, g1)
        matches = ismags.subgraph_isomorphisms_iter(symmetry=True)
        expected_symmetric = [{0: 0, 1: 1, 2: 2}, {0: 0, 1: 1, 3: 2}, {2: 0, 1: 1, 3: 2}]
        assert _matches_to_sets(matches) == _matches_to_sets(expected_symmetric)
        matches = ismags.subgraph_isomorphisms_iter(symmetry=False)
        expected_asymmetric = [{0: 2, 1: 1, 2: 0}, {0: 2, 1: 1, 3: 0}, {2: 2, 1: 1, 3: 0}]
        assert _matches_to_sets(matches) == _matches_to_sets(expected_symmetric + expected_asymmetric)

    def test_labeled_nodes(self):
        g1 = nx.Graph()
        nx.add_cycle(g1, range(3))
        g1.nodes[1]['attr'] = True
        g2 = g1.copy()
        g2.add_edge(1, 3)
        ismags = iso.ISMAGS(g2, g1, node_match=lambda x, y: x == y)
        matches = ismags.subgraph_isomorphisms_iter(symmetry=True)
        expected_symmetric = [{0: 0, 1: 1, 2: 2}]
        assert _matches_to_sets(matches) == _matches_to_sets(expected_symmetric)
        matches = ismags.subgraph_isomorphisms_iter(symmetry=False)
        expected_asymmetric = [{0: 2, 1: 1, 2: 0}]
        assert _matches_to_sets(matches) == _matches_to_sets(expected_symmetric + expected_asymmetric)

    def test_labeled_edges(self):
        g1 = nx.Graph()
        nx.add_cycle(g1, range(3))
        g1.edges[1, 2]['attr'] = True
        g2 = g1.copy()
        g2.add_edge(1, 3)
        ismags = iso.ISMAGS(g2, g1, edge_match=lambda x, y: x == y)
        matches = ismags.subgraph_isomorphisms_iter(symmetry=True)
        expected_symmetric = [{0: 0, 1: 1, 2: 2}]
        assert _matches_to_sets(matches) == _matches_to_sets(expected_symmetric)
        matches = ismags.subgraph_isomorphisms_iter(symmetry=False)
        expected_asymmetric = [{1: 2, 0: 0, 2: 1}]
        assert _matches_to_sets(matches) == _matches_to_sets(expected_symmetric + expected_asymmetric)