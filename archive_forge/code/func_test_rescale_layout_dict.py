import pytest
import networkx as nx
def test_rescale_layout_dict(self):
    G = nx.empty_graph()
    vpos = nx.random_layout(G, center=(1, 1))
    assert nx.rescale_layout_dict(vpos) == {}
    G = nx.empty_graph(2)
    vpos = {0: (0.0, 0.0), 1: (1.0, 1.0)}
    s_vpos = nx.rescale_layout_dict(vpos)
    assert np.linalg.norm([sum(x) for x in zip(*s_vpos.values())]) < 1e-06
    G = nx.empty_graph(3)
    vpos = {0: (0, 0), 1: (1, 1), 2: (0.5, 0.5)}
    s_vpos = nx.rescale_layout_dict(vpos)
    expectation = {0: np.array((-1, -1)), 1: np.array((1, 1)), 2: np.array((0, 0))}
    for k, v in expectation.items():
        assert (s_vpos[k] == v).all()
    s_vpos = nx.rescale_layout_dict(vpos, scale=2)
    expectation = {0: np.array((-2, -2)), 1: np.array((2, 2)), 2: np.array((0, 0))}
    for k, v in expectation.items():
        assert (s_vpos[k] == v).all()