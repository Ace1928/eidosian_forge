import pytest
import networkx as nx
def test_single_nodes(self):
    G = nx.path_graph(1)
    vpos = nx.shell_layout(G)
    assert not vpos[0].any()
    G = nx.path_graph(4)
    vpos = nx.shell_layout(G, [[0], [1, 2], [3]])
    assert not vpos[0].any()
    assert vpos[3].any()
    assert np.linalg.norm(vpos[3]) <= 1
    vpos = nx.shell_layout(G, [[0], [1, 2], [3]], rotate=0)
    assert np.linalg.norm(vpos[3]) <= 1