import pytest
import networkx as nx
def test_two_group_betweenness_value_zero(self):
    """
        Group betweenness centrality value of 0
        """
    G = nx.cycle_graph(7)
    C = [[0, 1, 6], [0, 1, 5]]
    b = nx.group_betweenness_centrality(G, C, weight=None, normalized=False)
    b_answer = [0.0, 3.0]
    assert b == b_answer