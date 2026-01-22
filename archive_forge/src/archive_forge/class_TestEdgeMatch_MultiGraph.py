import math
from operator import eq
import networkx as nx
import networkx.algorithms.isomorphism as iso
class TestEdgeMatch_MultiGraph:

    def setup_method(self):
        self.g1 = nx.MultiGraph()
        self.g2 = nx.MultiGraph()
        self.GM = iso.MultiGraphMatcher
        self.build()

    def build(self):
        g1 = self.g1
        g2 = self.g2
        g1.add_edge('A', 'B', color='green', weight=0, size=0.5)
        g1.add_edge('A', 'B', color='red', weight=1, size=0.35)
        g1.add_edge('A', 'B', color='red', weight=2, size=0.65)
        g2.add_edge('C', 'D', color='green', weight=1, size=0.5)
        g2.add_edge('C', 'D', color='red', weight=0, size=0.45)
        g2.add_edge('C', 'D', color='red', weight=2, size=0.65)
        if g1.is_multigraph():
            self.em = iso.numerical_multiedge_match('weight', 1)
            self.emc = iso.categorical_multiedge_match('color', '')
            self.emcm = iso.categorical_multiedge_match(['color', 'weight'], ['', 1])
            self.emg1 = iso.generic_multiedge_match('color', 'red', eq)
            self.emg2 = iso.generic_multiedge_match(['color', 'weight', 'size'], ['red', 1, 0.5], [eq, eq, math.isclose])
        else:
            self.em = iso.numerical_edge_match('weight', 1)
            self.emc = iso.categorical_edge_match('color', '')
            self.emcm = iso.categorical_edge_match(['color', 'weight'], ['', 1])
            self.emg1 = iso.generic_multiedge_match('color', 'red', eq)
            self.emg2 = iso.generic_edge_match(['color', 'weight', 'size'], ['red', 1, 0.5], [eq, eq, math.isclose])

    def test_weights_only(self):
        assert nx.is_isomorphic(self.g1, self.g2, edge_match=self.em)

    def test_colors_only(self):
        gm = self.GM(self.g1, self.g2, edge_match=self.emc)
        assert gm.is_isomorphic()

    def test_colorsandweights(self):
        gm = self.GM(self.g1, self.g2, edge_match=self.emcm)
        assert not gm.is_isomorphic()

    def test_generic1(self):
        gm = self.GM(self.g1, self.g2, edge_match=self.emg1)
        assert gm.is_isomorphic()

    def test_generic2(self):
        gm = self.GM(self.g1, self.g2, edge_match=self.emg2)
        assert not gm.is_isomorphic()