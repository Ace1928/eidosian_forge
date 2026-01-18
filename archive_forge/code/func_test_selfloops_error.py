import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_selfloops_error(self):
    G = nx.cycle_graph(4)
    G.add_edge(0, 0)
    pytest.raises(nx.NetworkXError, nx.inverse_line_graph, G)