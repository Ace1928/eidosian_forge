import networkx as nx
from networkx.utils import edges_equal
def test_nonisomorphic_trees(self):

    def f(x):
        return list(nx.nonisomorphic_trees(x))
    assert edges_equal(f(3)[0].edges(), [(0, 1), (0, 2)])
    assert edges_equal(f(4)[0].edges(), [(0, 1), (0, 3), (1, 2)])
    assert edges_equal(f(4)[1].edges(), [(0, 1), (0, 2), (0, 3)])