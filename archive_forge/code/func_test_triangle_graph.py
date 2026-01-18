import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_triangle_graph(self):
    G = nx.complete_graph(3)
    H = nx.inverse_line_graph(G)
    alternative_solution = nx.Graph()
    alternative_solution.add_edges_from([[0, 1], [0, 2], [0, 3]])
    assert nx.is_isomorphic(H, G) or nx.is_isomorphic(H, alternative_solution)