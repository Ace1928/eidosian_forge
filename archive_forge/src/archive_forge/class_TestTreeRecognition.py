import pytest
import networkx as nx
class TestTreeRecognition:
    graph = nx.Graph
    multigraph = nx.MultiGraph

    @classmethod
    def setup_class(cls):
        cls.T1 = cls.graph()
        cls.T2 = cls.graph()
        cls.T2.add_node(1)
        cls.T3 = cls.graph()
        cls.T3.add_nodes_from(range(5))
        edges = [(i, i + 1) for i in range(4)]
        cls.T3.add_edges_from(edges)
        cls.T5 = cls.multigraph()
        cls.T5.add_nodes_from(range(5))
        edges = [(i, i + 1) for i in range(4)]
        cls.T5.add_edges_from(edges)
        cls.T6 = cls.graph()
        cls.T6.add_nodes_from([6, 7])
        cls.T6.add_edge(6, 7)
        cls.F1 = nx.compose(cls.T6, cls.T3)
        cls.N4 = cls.graph()
        cls.N4.add_node(1)
        cls.N4.add_edge(1, 1)
        cls.N5 = cls.graph()
        cls.N5.add_nodes_from(range(5))
        cls.N6 = cls.graph()
        cls.N6.add_nodes_from(range(3))
        cls.N6.add_edges_from([(0, 1), (1, 2), (2, 0)])
        cls.NF1 = nx.compose(cls.T6, cls.N6)

    def test_null_tree(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.is_tree(self.graph())

    def test_null_tree2(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.is_tree(self.multigraph())

    def test_null_forest(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.is_forest(self.graph())

    def test_null_forest2(self):
        with pytest.raises(nx.NetworkXPointlessConcept):
            nx.is_forest(self.multigraph())

    def test_is_tree(self):
        assert nx.is_tree(self.T2)
        assert nx.is_tree(self.T3)
        assert nx.is_tree(self.T5)

    def test_is_not_tree(self):
        assert not nx.is_tree(self.N4)
        assert not nx.is_tree(self.N5)
        assert not nx.is_tree(self.N6)

    def test_is_forest(self):
        assert nx.is_forest(self.T2)
        assert nx.is_forest(self.T3)
        assert nx.is_forest(self.T5)
        assert nx.is_forest(self.F1)
        assert nx.is_forest(self.N5)

    def test_is_not_forest(self):
        assert not nx.is_forest(self.N4)
        assert not nx.is_forest(self.N6)
        assert not nx.is_forest(self.NF1)