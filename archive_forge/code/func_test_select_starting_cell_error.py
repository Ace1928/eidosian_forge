import pytest
import networkx as nx
from networkx.generators import line
from networkx.utils import edges_equal
def test_select_starting_cell_error(self):
    G = nx.diamond_graph()
    pytest.raises(nx.NetworkXError, line._select_starting_cell, G, (4, 0))
    pytest.raises(nx.NetworkXError, line._select_starting_cell, G, (0, 3))