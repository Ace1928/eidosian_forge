import math
from operator import eq
import networkx as nx
import networkx.algorithms.isomorphism as iso
class TestEdgeMatch_DiGraph(TestNodeMatch_Graph):

    def setup_method(self):
        TestNodeMatch_Graph.setup_method(self)
        self.g1 = nx.DiGraph()
        self.g2 = nx.DiGraph()
        self.build()