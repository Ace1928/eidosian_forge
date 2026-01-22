import gc
import pickle
import platform
import weakref
import pytest
import networkx as nx
from networkx.utils import edges_equal, graphs_equal, nodes_equal
class BaseAttrGraphTester(BaseGraphTester):
    """Tests of graph class attribute features."""

    def test_weighted_degree(self):
        G = self.Graph()
        G.add_edge(1, 2, weight=2, other=3)
        G.add_edge(2, 3, weight=3, other=4)
        assert sorted((d for n, d in G.degree(weight='weight'))) == [2, 3, 5]
        assert dict(G.degree(weight='weight')) == {1: 2, 2: 5, 3: 3}
        assert G.degree(1, weight='weight') == 2
        assert nodes_equal(G.degree([1], weight='weight'), [(1, 2)])
        assert nodes_equal((d for n, d in G.degree(weight='other')), [3, 7, 4])
        assert dict(G.degree(weight='other')) == {1: 3, 2: 7, 3: 4}
        assert G.degree(1, weight='other') == 3
        assert edges_equal(G.degree([1], weight='other'), [(1, 3)])

    def add_attributes(self, G):
        G.graph['foo'] = []
        G.nodes[0]['foo'] = []
        G.remove_edge(1, 2)
        ll = []
        G.add_edge(1, 2, foo=ll)
        G.add_edge(2, 1, foo=ll)

    def test_name(self):
        G = self.Graph(name='')
        assert G.name == ''
        G = self.Graph(name='test')
        assert G.name == 'test'

    def test_str_unnamed(self):
        G = self.Graph()
        G.add_edges_from([(1, 2), (2, 3)])
        assert str(G) == f'{type(G).__name__} with 3 nodes and 2 edges'

    def test_str_named(self):
        G = self.Graph(name='foo')
        G.add_edges_from([(1, 2), (2, 3)])
        assert str(G) == f"{type(G).__name__} named 'foo' with 3 nodes and 2 edges"

    def test_graph_chain(self):
        G = self.Graph([(0, 1), (1, 2)])
        DG = G.to_directed(as_view=True)
        SDG = DG.subgraph([0, 1])
        RSDG = SDG.reverse(copy=False)
        assert G is DG._graph
        assert DG is SDG._graph
        assert SDG is RSDG._graph

    def test_copy(self):
        G = self.Graph()
        G.add_node(0)
        G.add_edge(1, 2)
        self.add_attributes(G)
        H = G.copy()
        self.graphs_equal(H, G)
        self.different_attrdict(H, G)
        self.shallow_copy_attrdict(H, G)

    def test_class_copy(self):
        G = self.Graph()
        G.add_node(0)
        G.add_edge(1, 2)
        self.add_attributes(G)
        H = G.__class__(G)
        self.graphs_equal(H, G)
        self.different_attrdict(H, G)
        self.shallow_copy_attrdict(H, G)

    def test_fresh_copy(self):
        G = self.Graph()
        G.add_node(0)
        G.add_edge(1, 2)
        self.add_attributes(G)
        H = G.__class__()
        H.add_nodes_from(G)
        H.add_edges_from(G.edges())
        assert len(G.nodes[0]) == 1
        ddict = G.adj[1][2][0] if G.is_multigraph() else G.adj[1][2]
        assert len(ddict) == 1
        assert len(H.nodes[0]) == 0
        ddict = H.adj[1][2][0] if H.is_multigraph() else H.adj[1][2]
        assert len(ddict) == 0

    def is_deepcopy(self, H, G):
        self.graphs_equal(H, G)
        self.different_attrdict(H, G)
        self.deep_copy_attrdict(H, G)

    def deep_copy_attrdict(self, H, G):
        self.deepcopy_graph_attr(H, G)
        self.deepcopy_node_attr(H, G)
        self.deepcopy_edge_attr(H, G)

    def deepcopy_graph_attr(self, H, G):
        assert G.graph['foo'] == H.graph['foo']
        G.graph['foo'].append(1)
        assert G.graph['foo'] != H.graph['foo']

    def deepcopy_node_attr(self, H, G):
        assert G.nodes[0]['foo'] == H.nodes[0]['foo']
        G.nodes[0]['foo'].append(1)
        assert G.nodes[0]['foo'] != H.nodes[0]['foo']

    def deepcopy_edge_attr(self, H, G):
        assert G[1][2]['foo'] == H[1][2]['foo']
        G[1][2]['foo'].append(1)
        assert G[1][2]['foo'] != H[1][2]['foo']

    def is_shallow_copy(self, H, G):
        self.graphs_equal(H, G)
        self.shallow_copy_attrdict(H, G)

    def shallow_copy_attrdict(self, H, G):
        self.shallow_copy_graph_attr(H, G)
        self.shallow_copy_node_attr(H, G)
        self.shallow_copy_edge_attr(H, G)

    def shallow_copy_graph_attr(self, H, G):
        assert G.graph['foo'] == H.graph['foo']
        G.graph['foo'].append(1)
        assert G.graph['foo'] == H.graph['foo']

    def shallow_copy_node_attr(self, H, G):
        assert G.nodes[0]['foo'] == H.nodes[0]['foo']
        G.nodes[0]['foo'].append(1)
        assert G.nodes[0]['foo'] == H.nodes[0]['foo']

    def shallow_copy_edge_attr(self, H, G):
        assert G[1][2]['foo'] == H[1][2]['foo']
        G[1][2]['foo'].append(1)
        assert G[1][2]['foo'] == H[1][2]['foo']

    def same_attrdict(self, H, G):
        old_foo = H[1][2]['foo']
        H.adj[1][2]['foo'] = 'baz'
        assert G.edges == H.edges
        H.adj[1][2]['foo'] = old_foo
        assert G.edges == H.edges
        old_foo = H.nodes[0]['foo']
        H.nodes[0]['foo'] = 'baz'
        assert G.nodes == H.nodes
        H.nodes[0]['foo'] = old_foo
        assert G.nodes == H.nodes

    def different_attrdict(self, H, G):
        old_foo = H[1][2]['foo']
        H.adj[1][2]['foo'] = 'baz'
        assert G._adj != H._adj
        H.adj[1][2]['foo'] = old_foo
        assert G._adj == H._adj
        old_foo = H.nodes[0]['foo']
        H.nodes[0]['foo'] = 'baz'
        assert G._node != H._node
        H.nodes[0]['foo'] = old_foo
        assert G._node == H._node

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

    def test_graph_attr(self):
        G = self.K3.copy()
        G.graph['foo'] = 'bar'
        assert isinstance(G.graph, G.graph_attr_dict_factory)
        assert G.graph['foo'] == 'bar'
        del G.graph['foo']
        assert G.graph == {}
        H = self.Graph(foo='bar')
        assert H.graph['foo'] == 'bar'

    def test_node_attr(self):
        G = self.K3.copy()
        G.add_node(1, foo='bar')
        assert all((isinstance(d, G.node_attr_dict_factory) for u, d in G.nodes(data=True)))
        assert nodes_equal(G.nodes(), [0, 1, 2])
        assert nodes_equal(G.nodes(data=True), [(0, {}), (1, {'foo': 'bar'}), (2, {})])
        G.nodes[1]['foo'] = 'baz'
        assert nodes_equal(G.nodes(data=True), [(0, {}), (1, {'foo': 'baz'}), (2, {})])
        assert nodes_equal(G.nodes(data='foo'), [(0, None), (1, 'baz'), (2, None)])
        assert nodes_equal(G.nodes(data='foo', default='bar'), [(0, 'bar'), (1, 'baz'), (2, 'bar')])

    def test_node_attr2(self):
        G = self.K3.copy()
        a = {'foo': 'bar'}
        G.add_node(3, **a)
        assert nodes_equal(G.nodes(), [0, 1, 2, 3])
        assert nodes_equal(G.nodes(data=True), [(0, {}), (1, {}), (2, {}), (3, {'foo': 'bar'})])

    def test_edge_lookup(self):
        G = self.Graph()
        G.add_edge(1, 2, foo='bar')
        assert edges_equal(G.edges[1, 2], {'foo': 'bar'})

    def test_edge_attr(self):
        G = self.Graph()
        G.add_edge(1, 2, foo='bar')
        assert all((isinstance(d, G.edge_attr_dict_factory) for u, v, d in G.edges(data=True)))
        assert edges_equal(G.edges(data=True), [(1, 2, {'foo': 'bar'})])
        assert edges_equal(G.edges(data='foo'), [(1, 2, 'bar')])

    def test_edge_attr2(self):
        G = self.Graph()
        G.add_edges_from([(1, 2), (3, 4)], foo='foo')
        assert edges_equal(G.edges(data=True), [(1, 2, {'foo': 'foo'}), (3, 4, {'foo': 'foo'})])
        assert edges_equal(G.edges(data='foo'), [(1, 2, 'foo'), (3, 4, 'foo')])

    def test_edge_attr3(self):
        G = self.Graph()
        G.add_edges_from([(1, 2, {'weight': 32}), (3, 4, {'weight': 64})], foo='foo')
        assert edges_equal(G.edges(data=True), [(1, 2, {'foo': 'foo', 'weight': 32}), (3, 4, {'foo': 'foo', 'weight': 64})])
        G.remove_edges_from([(1, 2), (3, 4)])
        G.add_edge(1, 2, data=7, spam='bar', bar='foo')
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 7, 'spam': 'bar', 'bar': 'foo'})])

    def test_edge_attr4(self):
        G = self.Graph()
        G.add_edge(1, 2, data=7, spam='bar', bar='foo')
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 7, 'spam': 'bar', 'bar': 'foo'})])
        G[1][2]['data'] = 10
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 10, 'spam': 'bar', 'bar': 'foo'})])
        G.adj[1][2]['data'] = 20
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 20, 'spam': 'bar', 'bar': 'foo'})])
        G.edges[1, 2]['data'] = 21
        assert edges_equal(G.edges(data=True), [(1, 2, {'data': 21, 'spam': 'bar', 'bar': 'foo'})])
        G.adj[1][2]['listdata'] = [20, 200]
        G.adj[1][2]['weight'] = 20
        dd = {'data': 21, 'spam': 'bar', 'bar': 'foo', 'listdata': [20, 200], 'weight': 20}
        assert edges_equal(G.edges(data=True), [(1, 2, dd)])

    def test_to_undirected(self):
        G = self.K3
        self.add_attributes(G)
        H = nx.Graph(G)
        self.is_shallow_copy(H, G)
        self.different_attrdict(H, G)
        H = G.to_undirected()
        self.is_deepcopy(H, G)

    def test_to_directed_as_view(self):
        H = nx.path_graph(2, create_using=self.Graph)
        H2 = H.to_directed(as_view=True)
        assert H is H2._graph
        assert H2.has_edge(0, 1)
        assert H2.has_edge(1, 0) or H.is_directed()
        pytest.raises(nx.NetworkXError, H2.add_node, -1)
        pytest.raises(nx.NetworkXError, H2.add_edge, 1, 2)
        H.add_edge(1, 2)
        assert H2.has_edge(1, 2)
        assert H2.has_edge(2, 1) or H.is_directed()

    def test_to_undirected_as_view(self):
        H = nx.path_graph(2, create_using=self.Graph)
        H2 = H.to_undirected(as_view=True)
        assert H is H2._graph
        assert H2.has_edge(0, 1)
        assert H2.has_edge(1, 0)
        pytest.raises(nx.NetworkXError, H2.add_node, -1)
        pytest.raises(nx.NetworkXError, H2.add_edge, 1, 2)
        H.add_edge(1, 2)
        assert H2.has_edge(1, 2)
        assert H2.has_edge(2, 1)

    def test_directed_class(self):
        G = self.Graph()

        class newGraph(G.to_undirected_class()):

            def to_directed_class(self):
                return newDiGraph

            def to_undirected_class(self):
                return newGraph

        class newDiGraph(G.to_directed_class()):

            def to_directed_class(self):
                return newDiGraph

            def to_undirected_class(self):
                return newGraph
        G = newDiGraph() if G.is_directed() else newGraph()
        H = G.to_directed()
        assert isinstance(H, newDiGraph)
        H = G.to_undirected()
        assert isinstance(H, newGraph)

    def test_to_directed(self):
        G = self.K3
        self.add_attributes(G)
        H = nx.DiGraph(G)
        self.is_shallow_copy(H, G)
        self.different_attrdict(H, G)
        H = G.to_directed()
        self.is_deepcopy(H, G)

    def test_subgraph(self):
        G = self.K3
        self.add_attributes(G)
        H = G.subgraph([0, 1, 2, 5])
        self.graphs_equal(H, G)
        self.same_attrdict(H, G)
        self.shallow_copy_attrdict(H, G)
        H = G.subgraph(0)
        assert H.adj == {0: {}}
        H = G.subgraph([])
        assert H.adj == {}
        assert G.adj != {}

    def test_selfloops_attr(self):
        G = self.K3.copy()
        G.add_edge(0, 0)
        G.add_edge(1, 1, weight=2)
        assert edges_equal(nx.selfloop_edges(G, data=True), [(0, 0, {}), (1, 1, {'weight': 2})])
        assert edges_equal(nx.selfloop_edges(G, data='weight'), [(0, 0, None), (1, 1, 2)])