import pytest
import networkx as nx
def test_prominent_group_normalized_endpoints(self):
    """
        Prominent group with normalized result, with endpoints
        """
    G = nx.cycle_graph(7)
    k = 2
    b, g = nx.prominent_group(G, k, normalized=True, endpoints=True)
    b_answer, g_answer = (1.7, [2, 5])
    assert b == b_answer and g == g_answer