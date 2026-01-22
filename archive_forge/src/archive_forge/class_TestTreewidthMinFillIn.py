import itertools
import networkx as nx
from networkx.algorithms.approximation import (
from networkx.algorithms.approximation.treewidth import (
class TestTreewidthMinFillIn:
    """Unit tests for the treewidth_min_fill_in function."""

    @classmethod
    def setup_class(cls):
        """Setup for different kinds of trees"""
        cls.complete = nx.Graph()
        cls.complete.add_edge(1, 2)
        cls.complete.add_edge(2, 3)
        cls.complete.add_edge(1, 3)
        cls.small_tree = nx.Graph()
        cls.small_tree.add_edge(1, 2)
        cls.small_tree.add_edge(2, 3)
        cls.small_tree.add_edge(3, 4)
        cls.small_tree.add_edge(1, 4)
        cls.small_tree.add_edge(2, 4)
        cls.small_tree.add_edge(4, 5)
        cls.small_tree.add_edge(5, 6)
        cls.small_tree.add_edge(5, 7)
        cls.small_tree.add_edge(6, 7)
        cls.deterministic_graph = nx.Graph()
        cls.deterministic_graph.add_edge(1, 2)
        cls.deterministic_graph.add_edge(1, 3)
        cls.deterministic_graph.add_edge(3, 4)
        cls.deterministic_graph.add_edge(2, 4)
        cls.deterministic_graph.add_edge(3, 5)
        cls.deterministic_graph.add_edge(4, 5)
        cls.deterministic_graph.add_edge(3, 6)
        cls.deterministic_graph.add_edge(5, 6)

    def test_petersen_graph(self):
        """Test Petersen graph tree decomposition result"""
        G = nx.petersen_graph()
        _, decomp = treewidth_min_fill_in(G)
        is_tree_decomp(G, decomp)

    def test_small_tree_treewidth(self):
        """Test if the computed treewidth of the known self.small_tree is 2"""
        G = self.small_tree
        treewidth, _ = treewidth_min_fill_in(G)
        assert treewidth == 2

    def test_heuristic_abort(self):
        """Test if min_fill_in returns None for fully connected graph"""
        graph = {}
        for u in self.complete:
            graph[u] = set()
            for v in self.complete[u]:
                if u != v:
                    graph[u].add(v)
        next_node = min_fill_in_heuristic(graph)
        if next_node is None:
            pass
        else:
            assert False

    def test_empty_graph(self):
        """Test empty graph"""
        G = nx.Graph()
        _, _ = treewidth_min_fill_in(G)

    def test_two_component_graph(self):
        G = nx.Graph()
        G.add_node(1)
        G.add_node(2)
        treewidth, _ = treewidth_min_fill_in(G)
        assert treewidth == 0

    def test_not_sortable_nodes(self):
        G = nx.Graph([(0, 'a')])
        treewidth_min_fill_in(G)

    def test_heuristic_first_steps(self):
        """Test first steps of min_fill_in heuristic"""
        graph = {n: set(self.deterministic_graph[n]) - {n} for n in self.deterministic_graph}
        print(f'Graph {graph}:')
        elim_node = min_fill_in_heuristic(graph)
        steps = []
        while elim_node is not None:
            print(f'Removing {elim_node}:')
            steps.append(elim_node)
            nbrs = graph[elim_node]
            for u, v in itertools.permutations(nbrs, 2):
                if v not in graph[u]:
                    graph[u].add(v)
            for u in graph:
                if elim_node in graph[u]:
                    graph[u].remove(elim_node)
            del graph[elim_node]
            print(f'Graph {graph}:')
            elim_node = min_fill_in_heuristic(graph)
        assert steps[:2] == [6, 5]