import pytest
import networkx as nx
def test_spring_fixed_without_pos(self):
    G = nx.path_graph(4)
    pytest.raises(ValueError, nx.spring_layout, G, fixed=[0])
    pos = {0: (1, 1), 2: (0, 0)}
    pytest.raises(ValueError, nx.spring_layout, G, fixed=[0, 1], pos=pos)
    nx.spring_layout(G, fixed=[0, 2], pos=pos)