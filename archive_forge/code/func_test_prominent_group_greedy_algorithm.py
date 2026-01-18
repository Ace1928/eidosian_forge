import pytest
import networkx as nx
def test_prominent_group_greedy_algorithm(self):
    """
        Group betweenness centrality in a greedy algorithm
        """
    G = nx.cycle_graph(7)
    k = 2
    b, g = nx.prominent_group(G, k, normalized=True, endpoints=True, greedy=True)
    b_answer, g_answer = (1.7, [6, 3])
    assert b == b_answer and g == g_answer