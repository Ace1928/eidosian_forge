import pytest
import networkx as nx
def test_prominent_group_single_node(self):
    """
        Prominent group for single node
        """
    G = nx.path_graph(5)
    k = 1
    b, g = nx.prominent_group(G, k, normalized=False, endpoints=False)
    b_answer, g_answer = (4.0, [2])
    assert b == b_answer and g == g_answer