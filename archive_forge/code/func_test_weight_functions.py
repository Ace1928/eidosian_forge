import pytest
import networkx as nx
from networkx.utils import pairwise
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