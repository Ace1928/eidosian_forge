import itertools as it
import pytest
import networkx as nx
from networkx import vf2pp_is_isomorphic, vf2pp_isomorphism
class TestDiGraphISOVF2pp:

    def test_wikipedia_graph(self):
        edges1 = [(1, 5), (1, 2), (1, 4), (3, 2), (6, 2), (3, 4), (7, 3), (4, 8), (5, 8), (6, 5), (6, 7), (7, 8)]
        mapped = {1: 'a', 2: 'h', 3: 'd', 4: 'i', 5: 'g', 6: 'b', 7: 'j', 8: 'c'}
        G1 = nx.DiGraph(edges1)
        G2 = nx.relabel_nodes(G1, mapped)
        assert vf2pp_isomorphism(G1, G2) == mapped
        G1.remove_edge(1, 5)
        G1.add_edge(5, 1)
        assert vf2pp_isomorphism(G1, G2) is None

    def test_non_isomorphic_same_degree_sequence(self):
        """
                G1                           G2
        x--------------x              x--------------x
        | \\            |              | \\            |
        |  x-------x   |              |  x-------x   |
        |  |       |   |              |  |       |   |
        |  x-------x   |              |  x-------x   |
        | /            |              |            \\ |
        x--------------x              x--------------x
        """
        edges1 = [(1, 5), (1, 2), (4, 1), (3, 2), (3, 4), (4, 8), (5, 8), (6, 5), (6, 7), (7, 8)]
        edges2 = [(1, 5), (1, 2), (4, 1), (3, 2), (4, 3), (5, 8), (6, 5), (6, 7), (3, 7), (8, 7)]
        G1 = nx.DiGraph(edges1)
        G2 = nx.DiGraph(edges2)
        assert vf2pp_isomorphism(G1, G2) is None