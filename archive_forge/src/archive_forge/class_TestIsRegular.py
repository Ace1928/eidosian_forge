import pytest
import networkx
import networkx as nx
import networkx.algorithms.regular as reg
import networkx.generators as gen
class TestIsRegular:

    def test_is_regular1(self):
        g = gen.cycle_graph(4)
        assert reg.is_regular(g)

    def test_is_regular2(self):
        g = gen.complete_graph(5)
        assert reg.is_regular(g)

    def test_is_regular3(self):
        g = gen.lollipop_graph(5, 5)
        assert not reg.is_regular(g)

    def test_is_regular4(self):
        g = nx.DiGraph()
        g.add_edges_from([(0, 1), (1, 2), (2, 0)])
        assert reg.is_regular(g)