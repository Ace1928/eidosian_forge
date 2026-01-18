import pytest
import networkx as nx
def test_prominent_group_node_not_in_graph(self):
    """
        Node(s) in C not in graph, raises NodeNotFound exception
        """
    with pytest.raises(nx.NodeNotFound):
        nx.prominent_group(nx.path_graph(5), 1, C=[10])