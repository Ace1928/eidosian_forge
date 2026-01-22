import pytest
import networkx as nx
from networkx.utils import edges_equal
class TestInducedSubGraph:

    @classmethod
    def setup_class(cls):
        cls.K3 = G = nx.complete_graph(3)
        G.graph['foo'] = []
        G.nodes[0]['foo'] = []
        G.remove_edge(1, 2)
        ll = []
        G.add_edge(1, 2, foo=ll)
        G.add_edge(2, 1, foo=ll)

    def test_full_graph(self):
        G = self.K3
        H = nx.induced_subgraph(G, [0, 1, 2, 5])
        assert H.name == G.name
        self.graphs_equal(H, G)
        self.same_attrdict(H, G)

    def test_partial_subgraph(self):
        G = self.K3
        H = nx.induced_subgraph(G, 0)
        assert dict(H.adj) == {0: {}}
        assert dict(G.adj) != {0: {}}
        H = nx.induced_subgraph(G, [0, 1])
        assert dict(H.adj) == {0: {1: {}}, 1: {0: {}}}

    def same_attrdict(self, H, G):
        old_foo = H[1][2]['foo']
        H.edges[1, 2]['foo'] = 'baz'
        assert G.edges == H.edges
        H.edges[1, 2]['foo'] = old_foo
        assert G.edges == H.edges
        old_foo = H.nodes[0]['foo']
        H.nodes[0]['foo'] = 'baz'
        assert G.nodes == H.nodes
        H.nodes[0]['foo'] = old_foo
        assert G.nodes == H.nodes

    def graphs_equal(self, H, G):
        assert G._adj == H._adj
        assert G._node == H._node
        assert G.graph == H.graph
        assert G.name == H.name
        if not G.is_directed() and (not H.is_directed()):
            assert H._adj[1][2] is H._adj[2][1]
            assert G._adj[1][2] is G._adj[2][1]
        else:
            if not G.is_directed():
                G._pred = G._adj
                G._succ = G._adj
            if not H.is_directed():
                H._pred = H._adj
                H._succ = H._adj
            assert G._pred == H._pred
            assert G._succ == H._succ
            assert H._succ[1][2] is H._pred[2][1]
            assert G._succ[1][2] is G._pred[2][1]