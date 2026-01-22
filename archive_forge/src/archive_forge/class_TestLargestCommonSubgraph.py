import pytest
import networkx as nx
from networkx.algorithms import isomorphism as iso
class TestLargestCommonSubgraph:

    def test_mcis(self):
        graph1 = nx.Graph()
        graph1.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (4, 5)])
        graph1.nodes[1]['color'] = 0
        graph2 = nx.Graph()
        graph2.add_edges_from([(1, 2), (2, 3), (2, 4), (3, 4), (3, 5), (5, 6), (5, 7), (6, 7)])
        graph2.nodes[1]['color'] = 1
        graph2.nodes[6]['color'] = 2
        graph2.nodes[7]['color'] = 2
        ismags = iso.ISMAGS(graph1, graph2, node_match=iso.categorical_node_match('color', None))
        assert list(ismags.subgraph_isomorphisms_iter(True)) == []
        assert list(ismags.subgraph_isomorphisms_iter(False)) == []
        found_mcis = _matches_to_sets(ismags.largest_common_subgraph())
        expected = _matches_to_sets([{2: 2, 3: 4, 4: 3, 5: 5}, {2: 4, 3: 2, 4: 3, 5: 5}])
        assert expected == found_mcis
        ismags = iso.ISMAGS(graph2, graph1, node_match=iso.categorical_node_match('color', None))
        assert list(ismags.subgraph_isomorphisms_iter(True)) == []
        assert list(ismags.subgraph_isomorphisms_iter(False)) == []
        found_mcis = _matches_to_sets(ismags.largest_common_subgraph())
        expected = _matches_to_sets([{2: 2, 3: 4, 4: 3, 5: 5}, {4: 2, 2: 3, 3: 4, 5: 5}])
        assert expected == found_mcis

    def test_symmetry_mcis(self):
        graph1 = nx.Graph()
        nx.add_path(graph1, range(4))
        graph2 = nx.Graph()
        nx.add_path(graph2, range(3))
        graph2.add_edge(1, 3)
        ismags1 = iso.ISMAGS(graph1, graph2, node_match=iso.categorical_node_match('color', None))
        assert list(ismags1.subgraph_isomorphisms_iter(True)) == []
        found_mcis = _matches_to_sets(ismags1.largest_common_subgraph())
        expected = _matches_to_sets([{0: 0, 1: 1, 2: 2}, {1: 0, 3: 2, 2: 1}])
        assert expected == found_mcis
        ismags2 = iso.ISMAGS(graph2, graph1, node_match=iso.categorical_node_match('color', None))
        assert list(ismags2.subgraph_isomorphisms_iter(True)) == []
        found_mcis = _matches_to_sets(ismags2.largest_common_subgraph())
        expected = _matches_to_sets([{3: 2, 0: 0, 1: 1}, {2: 0, 0: 2, 1: 1}, {3: 0, 0: 2, 1: 1}, {3: 0, 1: 1, 2: 2}, {0: 0, 1: 1, 2: 2}, {2: 0, 3: 2, 1: 1}])
        assert expected == found_mcis
        found_mcis1 = _matches_to_sets(ismags1.largest_common_subgraph(False))
        found_mcis2 = ismags2.largest_common_subgraph(False)
        found_mcis2 = [{v: k for k, v in d.items()} for d in found_mcis2]
        found_mcis2 = _matches_to_sets(found_mcis2)
        expected = _matches_to_sets([{3: 2, 1: 3, 2: 1}, {2: 0, 0: 2, 1: 1}, {1: 2, 3: 3, 2: 1}, {3: 0, 1: 3, 2: 1}, {0: 2, 2: 3, 1: 1}, {3: 0, 1: 2, 2: 1}, {2: 0, 0: 3, 1: 1}, {0: 0, 2: 3, 1: 1}, {1: 0, 3: 3, 2: 1}, {1: 0, 3: 2, 2: 1}, {0: 3, 1: 1, 2: 2}, {0: 0, 1: 1, 2: 2}])
        assert expected == found_mcis1
        assert expected == found_mcis2