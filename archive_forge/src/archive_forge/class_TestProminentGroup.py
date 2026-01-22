import pytest
import networkx as nx
class TestProminentGroup:
    np = pytest.importorskip('numpy')
    pd = pytest.importorskip('pandas')

    def test_prominent_group_single_node(self):
        """
        Prominent group for single node
        """
        G = nx.path_graph(5)
        k = 1
        b, g = nx.prominent_group(G, k, normalized=False, endpoints=False)
        b_answer, g_answer = (4.0, [2])
        assert b == b_answer and g == g_answer

    def test_prominent_group_with_c(self):
        """
        Prominent group without some nodes
        """
        G = nx.path_graph(5)
        k = 1
        b, g = nx.prominent_group(G, k, normalized=False, C=[2])
        b_answer, g_answer = (3.0, [1])
        assert b == b_answer and g == g_answer

    def test_prominent_group_normalized_endpoints(self):
        """
        Prominent group with normalized result, with endpoints
        """
        G = nx.cycle_graph(7)
        k = 2
        b, g = nx.prominent_group(G, k, normalized=True, endpoints=True)
        b_answer, g_answer = (1.7, [2, 5])
        assert b == b_answer and g == g_answer

    def test_prominent_group_disconnected_graph(self):
        """
        Prominent group of disconnected graph
        """
        G = nx.path_graph(6)
        G.remove_edge(0, 1)
        k = 1
        b, g = nx.prominent_group(G, k, weight=None, normalized=False)
        b_answer, g_answer = (4.0, [3])
        assert b == b_answer and g == g_answer

    def test_prominent_group_node_not_in_graph(self):
        """
        Node(s) in C not in graph, raises NodeNotFound exception
        """
        with pytest.raises(nx.NodeNotFound):
            nx.prominent_group(nx.path_graph(5), 1, C=[10])

    def test_group_betweenness_directed_weighted(self):
        """
        Group betweenness centrality in a directed and weighted graph
        """
        G = nx.DiGraph()
        G.add_edge(1, 0, weight=1)
        G.add_edge(0, 2, weight=2)
        G.add_edge(1, 2, weight=3)
        G.add_edge(3, 1, weight=4)
        G.add_edge(2, 3, weight=1)
        G.add_edge(4, 3, weight=6)
        G.add_edge(2, 4, weight=7)
        k = 2
        b, g = nx.prominent_group(G, k, weight='weight', normalized=False)
        b_answer, g_answer = (5.0, [1, 2])
        assert b == b_answer and g == g_answer

    def test_prominent_group_greedy_algorithm(self):
        """
        Group betweenness centrality in a greedy algorithm
        """
        G = nx.cycle_graph(7)
        k = 2
        b, g = nx.prominent_group(G, k, normalized=True, endpoints=True, greedy=True)
        b_answer, g_answer = (1.7, [6, 3])
        assert b == b_answer and g == g_answer