import networkx as nx
from .test_digraph import BaseDiGraphTester
from .test_digraph import TestDiGraph as _TestDiGraph
from .test_graph import BaseGraphTester
from .test_graph import TestGraph as _TestGraph
from .test_multidigraph import TestMultiDiGraph as _TestMultiDiGraph
from .test_multigraph import TestMultiGraph as _TestMultiGraph
class TestSpecialMultiGraph(_TestMultiGraph):

    def setup_method(self):
        _TestMultiGraph.setup_method(self)
        self.Graph = nx.MultiGraph