import pytest
import networkx as nx
class TestUnDirectedDedensification:

    def build_original_graph(self):
        """
        Builds graph shown in the original research paper
        """
        original_matrix = [('1', 'CB'), ('2', 'ABC'), ('3', ['A', 'B', '6']), ('4', 'ABC'), ('5', 'AB'), ('6', ['5']), ('A', ['6'])]
        graph = nx.Graph()
        for source, targets in original_matrix:
            for target in targets:
                graph.add_edge(source, target)
        return graph

    def test_empty(self):
        """
        Verify that an empty undirected graph results in no compressor nodes
        """
        G = nx.Graph()
        compressed_G, c_nodes = nx.dedensify(G, threshold=2)
        assert c_nodes == set()

    def setup_method(self):
        self.c_nodes = ('6AB', 'ABC')

    def build_compressed_graph(self):
        compressed_matrix = [('1', ['B', 'C']), ('2', ['ABC']), ('3', ['6AB']), ('4', ['ABC']), ('5', ['6AB']), ('6', ['6AB', 'A']), ('A', ['6AB', 'ABC']), ('B', ['ABC', '6AB']), ('C', ['ABC'])]
        compressed_graph = nx.Graph()
        for source, targets in compressed_matrix:
            for target in targets:
                compressed_graph.add_edge(source, target)
        return compressed_graph

    def test_dedensify_edges(self):
        """
        Verifies that dedensify produced correct compressor nodes and the
        correct edges to/from the compressor nodes in an undirected graph
        """
        G = self.build_original_graph()
        c_G, c_nodes = nx.dedensify(G, threshold=2)
        v_compressed_G = self.build_compressed_graph()
        for s, t in c_G.edges():
            o_s = ''.join(sorted(s))
            o_t = ''.join(sorted(t))
            has_compressed_edge = c_G.has_edge(s, t)
            verified_has_compressed_edge = v_compressed_G.has_edge(o_s, o_t)
            assert has_compressed_edge == verified_has_compressed_edge
        assert len(c_nodes) == len(self.c_nodes)

    def test_dedensify_edge_count(self):
        """
        Verifies that dedensify produced the correct number of edges in an
        undirected graph
        """
        G = self.build_original_graph()
        c_G, c_nodes = nx.dedensify(G, threshold=2, copy=True)
        compressed_edge_count = len(c_G.edges())
        verified_original_edge_count = len(G.edges())
        assert compressed_edge_count <= verified_original_edge_count
        verified_compressed_G = self.build_compressed_graph()
        verified_compressed_edge_count = len(verified_compressed_G.edges())
        assert compressed_edge_count == verified_compressed_edge_count