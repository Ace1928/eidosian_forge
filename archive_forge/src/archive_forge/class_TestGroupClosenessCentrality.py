import pytest
import networkx as nx
class TestGroupClosenessCentrality:

    def test_group_closeness_single_node(self):
        """
        Group closeness centrality for a single node group
        """
        G = nx.path_graph(5)
        c = nx.group_closeness_centrality(G, [1])
        c_answer = nx.closeness_centrality(G, 1)
        assert c == c_answer

    def test_group_closeness_disconnected(self):
        """
        Group closeness centrality for a disconnected graph
        """
        G = nx.Graph()
        G.add_nodes_from([1, 2, 3, 4])
        c = nx.group_closeness_centrality(G, [1, 2])
        c_answer = 0
        assert c == c_answer

    def test_group_closeness_multiple_node(self):
        """
        Group closeness centrality for a group with more than
        1 node
        """
        G = nx.path_graph(4)
        c = nx.group_closeness_centrality(G, [1, 2])
        c_answer = 1
        assert c == c_answer

    def test_group_closeness_node_not_in_graph(self):
        """
        Node(s) in S not in graph, raises NodeNotFound exception
        """
        with pytest.raises(nx.NodeNotFound):
            nx.group_closeness_centrality(nx.path_graph(5), [6, 7, 8])