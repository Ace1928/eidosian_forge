import importlib.resources
import os
import random
import struct
import networkx as nx
from networkx.algorithms import isomorphism as iso
class TestVF2GraphDB:

    @staticmethod
    def create_graph(filename):
        """Creates a Graph instance from the filename."""
        fh = open(filename, mode='rb')
        nodes = struct.unpack('<H', fh.read(2))[0]
        graph = nx.Graph()
        for from_node in range(nodes):
            edges = struct.unpack('<H', fh.read(2))[0]
            for edge in range(edges):
                to_node = struct.unpack('<H', fh.read(2))[0]
                graph.add_edge(from_node, to_node)
        fh.close()
        return graph

    def test_graph(self):
        head = importlib.resources.files('networkx.algorithms.isomorphism.tests')
        g1 = self.create_graph(head / 'iso_r01_s80.A99')
        g2 = self.create_graph(head / 'iso_r01_s80.B99')
        gm = iso.GraphMatcher(g1, g2)
        assert gm.is_isomorphic()

    def test_subgraph(self):
        head = importlib.resources.files('networkx.algorithms.isomorphism.tests')
        subgraph = self.create_graph(head / 'si2_b06_m200.A99')
        graph = self.create_graph(head / 'si2_b06_m200.B99')
        gm = iso.GraphMatcher(graph, subgraph)
        assert gm.subgraph_is_isomorphic()
        assert gm.subgraph_is_monomorphic()