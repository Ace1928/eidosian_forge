import pytest
import networkx as nx
from networkx.utils import pairwise
class TestWeightedPath(WeightedTestBase):

    def test_dijkstra(self):
        D, P = nx.single_source_dijkstra(self.XG, 's')
        validate_path(self.XG, 's', 'v', 9, P['v'])
        assert D['v'] == 9
        validate_path(self.XG, 's', 'v', 9, nx.single_source_dijkstra_path(self.XG, 's')['v'])
        assert dict(nx.single_source_dijkstra_path_length(self.XG, 's'))['v'] == 9
        validate_path(self.XG, 's', 'v', 9, nx.single_source_dijkstra(self.XG, 's')[1]['v'])
        validate_path(self.MXG, 's', 'v', 9, nx.single_source_dijkstra_path(self.MXG, 's')['v'])
        GG = self.XG.to_undirected()
        GG['u']['x']['weight'] = 2
        D, P = nx.single_source_dijkstra(GG, 's')
        validate_path(GG, 's', 'v', 8, P['v'])
        assert D['v'] == 8
        validate_path(GG, 's', 'v', 8, nx.dijkstra_path(GG, 's', 'v'))
        assert nx.dijkstra_path_length(GG, 's', 'v') == 8
        validate_path(self.XG2, 1, 3, 4, nx.dijkstra_path(self.XG2, 1, 3))
        validate_path(self.XG3, 0, 3, 15, nx.dijkstra_path(self.XG3, 0, 3))
        assert nx.dijkstra_path_length(self.XG3, 0, 3) == 15
        validate_path(self.XG4, 0, 2, 4, nx.dijkstra_path(self.XG4, 0, 2))
        assert nx.dijkstra_path_length(self.XG4, 0, 2) == 4
        validate_path(self.MXG4, 0, 2, 4, nx.dijkstra_path(self.MXG4, 0, 2))
        validate_path(self.G, 's', 'v', 2, nx.single_source_dijkstra(self.G, 's', 'v')[1])
        validate_path(self.G, 's', 'v', 2, nx.single_source_dijkstra(self.G, 's')[1]['v'])
        validate_path(self.G, 's', 'v', 2, nx.dijkstra_path(self.G, 's', 'v'))
        assert nx.dijkstra_path_length(self.G, 's', 'v') == 2
        pytest.raises(nx.NetworkXNoPath, nx.dijkstra_path, self.G, 's', 'moon')
        pytest.raises(nx.NetworkXNoPath, nx.dijkstra_path_length, self.G, 's', 'moon')
        validate_path(self.cycle, 0, 3, 3, nx.dijkstra_path(self.cycle, 0, 3))
        validate_path(self.cycle, 0, 4, 3, nx.dijkstra_path(self.cycle, 0, 4))
        assert nx.single_source_dijkstra(self.cycle, 0, 0) == (0, [0])

    def test_bidirectional_dijkstra(self):
        validate_length_path(self.XG, 's', 'v', 9, *nx.bidirectional_dijkstra(self.XG, 's', 'v'))
        validate_length_path(self.G, 's', 'v', 2, *nx.bidirectional_dijkstra(self.G, 's', 'v'))
        validate_length_path(self.cycle, 0, 3, 3, *nx.bidirectional_dijkstra(self.cycle, 0, 3))
        validate_length_path(self.cycle, 0, 4, 3, *nx.bidirectional_dijkstra(self.cycle, 0, 4))
        validate_length_path(self.XG3, 0, 3, 15, *nx.bidirectional_dijkstra(self.XG3, 0, 3))
        validate_length_path(self.XG4, 0, 2, 4, *nx.bidirectional_dijkstra(self.XG4, 0, 2))
        P = nx.single_source_dijkstra_path(self.XG, 's')['v']
        validate_path(self.XG, 's', 'v', sum((self.XG[u][v]['weight'] for u, v in zip(P[:-1], P[1:]))), nx.dijkstra_path(self.XG, 's', 'v'))
        G = nx.path_graph(2)
        pytest.raises(nx.NodeNotFound, nx.bidirectional_dijkstra, G, 3, 0)

    def test_weight_functions(self):

        def heuristic(*z):
            return sum((val ** 2 for val in z))

        def getpath(pred, v, s):
            return [v] if v == s else getpath(pred, pred[v], s) + [v]

        def goldberg_radzik(g, s, t, weight='weight'):
            pred, dist = nx.goldberg_radzik(g, s, weight=weight)
            dist = dist[t]
            return (dist, getpath(pred, t, s))

        def astar(g, s, t, weight='weight'):
            path = nx.astar_path(g, s, t, heuristic, weight=weight)
            dist = nx.astar_path_length(g, s, t, heuristic, weight=weight)
            return (dist, path)

        def vlp(G, s, t, l, F, w):
            res = F(G, s, t, weight=w)
            validate_length_path(G, s, t, l, *res, weight=w)
        G = self.cycle
        s = 6
        t = 4
        path = [6] + list(range(t + 1))

        def weight(u, v, _):
            return 1 + v ** 2
        length = sum((weight(u, v, None) for u, v in pairwise(path)))
        vlp(G, s, t, length, nx.bidirectional_dijkstra, weight)
        vlp(G, s, t, length, nx.single_source_dijkstra, weight)
        vlp(G, s, t, length, nx.single_source_bellman_ford, weight)
        vlp(G, s, t, length, goldberg_radzik, weight)
        vlp(G, s, t, length, astar, weight)

        def weight(u, v, _):
            return 2 ** (u * v)
        length = sum((weight(u, v, None) for u, v in pairwise(path)))
        vlp(G, s, t, length, nx.bidirectional_dijkstra, weight)
        vlp(G, s, t, length, nx.single_source_dijkstra, weight)
        vlp(G, s, t, length, nx.single_source_bellman_ford, weight)
        vlp(G, s, t, length, goldberg_radzik, weight)
        vlp(G, s, t, length, astar, weight)

    def test_bidirectional_dijkstra_no_path(self):
        with pytest.raises(nx.NetworkXNoPath):
            G = nx.Graph()
            nx.add_path(G, [1, 2, 3])
            nx.add_path(G, [4, 5, 6])
            path = nx.bidirectional_dijkstra(G, 1, 6)

    @pytest.mark.parametrize('fn', (nx.dijkstra_path, nx.dijkstra_path_length, nx.single_source_dijkstra_path, nx.single_source_dijkstra_path_length, nx.single_source_dijkstra, nx.dijkstra_predecessor_and_distance))
    def test_absent_source(self, fn):
        G = nx.path_graph(2)
        with pytest.raises(nx.NodeNotFound):
            fn(G, 3, 0)
        with pytest.raises(nx.NodeNotFound):
            fn(G, 3, 3)

    def test_dijkstra_predecessor1(self):
        G = nx.path_graph(4)
        assert nx.dijkstra_predecessor_and_distance(G, 0) == ({0: [], 1: [0], 2: [1], 3: [2]}, {0: 0, 1: 1, 2: 2, 3: 3})

    def test_dijkstra_predecessor2(self):
        G = nx.Graph([(0, 1), (1, 2), (2, 3), (3, 0)])
        pred, dist = nx.dijkstra_predecessor_and_distance(G, 0)
        assert pred[0] == []
        assert pred[1] == [0]
        assert pred[2] in [[1, 3], [3, 1]]
        assert pred[3] == [0]
        assert dist == {0: 0, 1: 1, 2: 2, 3: 1}

    def test_dijkstra_predecessor3(self):
        XG = nx.DiGraph()
        XG.add_weighted_edges_from([('s', 'u', 10), ('s', 'x', 5), ('u', 'v', 1), ('u', 'x', 2), ('v', 'y', 1), ('x', 'u', 3), ('x', 'v', 5), ('x', 'y', 2), ('y', 's', 7), ('y', 'v', 6)])
        P, D = nx.dijkstra_predecessor_and_distance(XG, 's')
        assert P['v'] == ['u']
        assert D['v'] == 9
        P, D = nx.dijkstra_predecessor_and_distance(XG, 's', cutoff=8)
        assert 'v' not in D

    def test_single_source_dijkstra_path_length(self):
        pl = nx.single_source_dijkstra_path_length
        assert dict(pl(self.MXG4, 0))[2] == 4
        spl = pl(self.MXG4, 0, cutoff=2)
        assert 2 not in spl

    def test_bidirectional_dijkstra_multigraph(self):
        G = nx.MultiGraph()
        G.add_edge('a', 'b', weight=10)
        G.add_edge('a', 'b', weight=100)
        dp = nx.bidirectional_dijkstra(G, 'a', 'b')
        assert dp == (10, ['a', 'b'])

    def test_dijkstra_pred_distance_multigraph(self):
        G = nx.MultiGraph()
        G.add_edge('a', 'b', key='short', foo=5, weight=100)
        G.add_edge('a', 'b', key='long', bar=1, weight=110)
        p, d = nx.dijkstra_predecessor_and_distance(G, 'a')
        assert p == {'a': [], 'b': ['a']}
        assert d == {'a': 0, 'b': 100}

    def test_negative_edge_cycle(self):
        G = nx.cycle_graph(5, create_using=nx.DiGraph())
        assert not nx.negative_edge_cycle(G)
        G.add_edge(8, 9, weight=-7)
        G.add_edge(9, 8, weight=3)
        graph_size = len(G)
        assert nx.negative_edge_cycle(G)
        assert graph_size == len(G)
        pytest.raises(ValueError, nx.single_source_dijkstra_path_length, G, 8)
        pytest.raises(ValueError, nx.single_source_dijkstra, G, 8)
        pytest.raises(ValueError, nx.dijkstra_predecessor_and_distance, G, 8)
        G.add_edge(9, 10)
        pytest.raises(ValueError, nx.bidirectional_dijkstra, G, 8, 10)
        G = nx.MultiDiGraph()
        G.add_edge(2, 2, weight=-1)
        assert nx.negative_edge_cycle(G)

    def test_negative_edge_cycle_empty(self):
        G = nx.DiGraph()
        assert not nx.negative_edge_cycle(G)

    def test_negative_edge_cycle_custom_weight_key(self):
        d = nx.DiGraph()
        d.add_edge('a', 'b', w=-2)
        d.add_edge('b', 'a', w=-1)
        assert nx.negative_edge_cycle(d, weight='w')

    def test_weight_function(self):
        """Tests that a callable weight is interpreted as a weight
        function instead of an edge attribute.

        """
        G = nx.complete_graph(3)
        G.adj[0][2]['weight'] = 10
        G.adj[0][1]['weight'] = 1
        G.adj[1][2]['weight'] = 1

        def weight(u, v, d):
            return 1 / d['weight']
        distance, path = nx.single_source_dijkstra(G, 0, 2)
        assert distance == 2
        assert path == [0, 1, 2]
        distance, path = nx.single_source_dijkstra(G, 0, 2, weight=weight)
        assert distance == 1 / 10
        assert path == [0, 2]

    def test_all_pairs_dijkstra_path(self):
        cycle = nx.cycle_graph(7)
        p = dict(nx.all_pairs_dijkstra_path(cycle))
        assert p[0][3] == [0, 1, 2, 3]
        cycle[1][2]['weight'] = 10
        p = dict(nx.all_pairs_dijkstra_path(cycle))
        assert p[0][3] == [0, 6, 5, 4, 3]

    def test_all_pairs_dijkstra_path_length(self):
        cycle = nx.cycle_graph(7)
        pl = dict(nx.all_pairs_dijkstra_path_length(cycle))
        assert pl[0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        cycle[1][2]['weight'] = 10
        pl = dict(nx.all_pairs_dijkstra_path_length(cycle))
        assert pl[0] == {0: 0, 1: 1, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}

    def test_all_pairs_dijkstra(self):
        cycle = nx.cycle_graph(7)
        out = dict(nx.all_pairs_dijkstra(cycle))
        assert out[0][0] == {0: 0, 1: 1, 2: 2, 3: 3, 4: 3, 5: 2, 6: 1}
        assert out[0][1][3] == [0, 1, 2, 3]
        cycle[1][2]['weight'] = 10
        out = dict(nx.all_pairs_dijkstra(cycle))
        assert out[0][0] == {0: 0, 1: 1, 2: 5, 3: 4, 4: 3, 5: 2, 6: 1}
        assert out[0][1][3] == [0, 6, 5, 4, 3]