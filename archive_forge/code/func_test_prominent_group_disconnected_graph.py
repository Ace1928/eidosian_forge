import pytest
import networkx as nx
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