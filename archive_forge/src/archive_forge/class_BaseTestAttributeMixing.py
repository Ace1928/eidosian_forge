import networkx as nx
class BaseTestAttributeMixing:

    @classmethod
    def setup_class(cls):
        G = nx.Graph()
        G.add_nodes_from([0, 1], fish='one')
        G.add_nodes_from([2, 3], fish='two')
        G.add_nodes_from([4], fish='red')
        G.add_nodes_from([5], fish='blue')
        G.add_edges_from([(0, 1), (2, 3), (0, 4), (2, 5)])
        cls.G = G
        D = nx.DiGraph()
        D.add_nodes_from([0, 1], fish='one')
        D.add_nodes_from([2, 3], fish='two')
        D.add_nodes_from([4], fish='red')
        D.add_nodes_from([5], fish='blue')
        D.add_edges_from([(0, 1), (2, 3), (0, 4), (2, 5)])
        cls.D = D
        M = nx.MultiGraph()
        M.add_nodes_from([0, 1], fish='one')
        M.add_nodes_from([2, 3], fish='two')
        M.add_nodes_from([4], fish='red')
        M.add_nodes_from([5], fish='blue')
        M.add_edges_from([(0, 1), (0, 1), (2, 3)])
        cls.M = M
        S = nx.Graph()
        S.add_nodes_from([0, 1], fish='one')
        S.add_nodes_from([2, 3], fish='two')
        S.add_nodes_from([4], fish='red')
        S.add_nodes_from([5], fish='blue')
        S.add_edge(0, 0)
        S.add_edge(2, 2)
        cls.S = S
        N = nx.Graph()
        N.add_nodes_from([0, 1], margin=-2)
        N.add_nodes_from([2, 3], margin=-2)
        N.add_nodes_from([4], margin=-3)
        N.add_nodes_from([5], margin=-4)
        N.add_edges_from([(0, 1), (2, 3), (0, 4), (2, 5)])
        cls.N = N
        F = nx.Graph()
        F.add_edges_from([(0, 3), (1, 3), (2, 3)], weight=0.5)
        F.add_edge(0, 2, weight=1)
        nx.set_node_attributes(F, dict(F.degree(weight='weight')), 'margin')
        cls.F = F
        K = nx.Graph()
        K.add_nodes_from([1, 2], margin=-1)
        K.add_nodes_from([3], margin=1)
        K.add_nodes_from([4], margin=2)
        K.add_edges_from([(3, 4), (1, 2), (1, 3)])
        cls.K = K