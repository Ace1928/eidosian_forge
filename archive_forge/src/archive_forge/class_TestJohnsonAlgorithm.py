import pytest
import networkx as nx
from networkx.utils import pairwise
class TestJohnsonAlgorithm(WeightedTestBase):

    def test_single_node_graph(self):
        G = nx.DiGraph()
        G.add_node(0)
        assert nx.johnson(G) == {0: {0: [0]}}

    def test_negative_cycle(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from([('0', '3', 3), ('0', '1', -5), ('1', '0', -5), ('0', '2', 2), ('1', '2', 4), ('2', '3', 1)])
        pytest.raises(nx.NetworkXUnbounded, nx.johnson, G)
        G = nx.Graph()
        G.add_weighted_edges_from([('0', '3', 3), ('0', '1', -5), ('1', '0', -5), ('0', '2', 2), ('1', '2', 4), ('2', '3', 1)])
        pytest.raises(nx.NetworkXUnbounded, nx.johnson, G)

    def test_negative_weights(self):
        G = nx.DiGraph()
        G.add_weighted_edges_from([('0', '3', 3), ('0', '1', -5), ('0', '2', 2), ('1', '2', 4), ('2', '3', 1)])
        paths = nx.johnson(G)
        assert paths == {'1': {'1': ['1'], '3': ['1', '2', '3'], '2': ['1', '2']}, '0': {'1': ['0', '1'], '0': ['0'], '3': ['0', '1', '2', '3'], '2': ['0', '1', '2']}, '3': {'3': ['3']}, '2': {'3': ['2', '3'], '2': ['2']}}

    def test_unweighted_graph(self):
        G = nx.Graph()
        G.add_edges_from([(1, 0), (2, 1)])
        H = G.copy()
        nx.set_edge_attributes(H, values=1, name='weight')
        assert nx.johnson(G) == nx.johnson(H)

    def test_partially_weighted_graph_with_negative_edges(self):
        G = nx.DiGraph()
        G.add_edges_from([(0, 1), (1, 2), (2, 0), (1, 0)])
        G[1][0]['weight'] = -2
        G[0][1]['weight'] = 3
        G[1][2]['weight'] = -4
        H = G.copy()
        H[2][0]['weight'] = 1
        I = G.copy()
        I[2][0]['weight'] = 8
        assert nx.johnson(G) == nx.johnson(H)
        assert nx.johnson(G) != nx.johnson(I)

    def test_graphs(self):
        validate_path(self.XG, 's', 'v', 9, nx.johnson(self.XG)['s']['v'])
        validate_path(self.MXG, 's', 'v', 9, nx.johnson(self.MXG)['s']['v'])
        validate_path(self.XG2, 1, 3, 4, nx.johnson(self.XG2)[1][3])
        validate_path(self.XG3, 0, 3, 15, nx.johnson(self.XG3)[0][3])
        validate_path(self.XG4, 0, 2, 4, nx.johnson(self.XG4)[0][2])
        validate_path(self.MXG4, 0, 2, 4, nx.johnson(self.MXG4)[0][2])