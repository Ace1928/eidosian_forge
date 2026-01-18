import pytest
import networkx as nx
def test_spiral_layout_equidistant(self):
    G = nx.path_graph(10)
    pos = nx.spiral_layout(G, equidistant=True)
    p = np.array(list(pos.values()))
    dist = np.linalg.norm(p[1:] - p[:-1], axis=1)
    assert np.allclose(np.diff(dist), 0, atol=0.001)