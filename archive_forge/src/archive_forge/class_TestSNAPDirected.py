import pytest
import networkx as nx
class TestSNAPDirected(AbstractSNAP):

    def build_original_graph(self):
        nodes = {'A': {'color': 'Red'}, 'B': {'color': 'Red'}, 'C': {'color': 'Green'}, 'D': {'color': 'Green'}, 'E': {'color': 'Blue'}, 'F': {'color': 'Blue'}, 'G': {'color': 'Yellow'}, 'H': {'color': 'Yellow'}}
        edges = [('A', 'C', 'Strong'), ('A', 'E', 'Strong'), ('A', 'F', 'Weak'), ('B', 'D', 'Strong'), ('B', 'E', 'Weak'), ('B', 'F', 'Strong'), ('C', 'G', 'Strong'), ('C', 'F', 'Strong'), ('D', 'E', 'Strong'), ('D', 'H', 'Strong'), ('G', 'E', 'Strong'), ('H', 'F', 'Strong')]
        G = nx.DiGraph()
        for node in nodes:
            attributes = nodes[node]
            G.add_node(node, **attributes)
        for source, target, type in edges:
            G.add_edge(source, target, type=type)
        return G

    def build_summary_graph(self):
        nodes = {'Supernode-0': {'color': 'Red'}, 'Supernode-1': {'color': 'Green'}, 'Supernode-2': {'color': 'Blue'}, 'Supernode-3': {'color': 'Yellow'}}
        edges = [('Supernode-0', 'Supernode-1', [{'type': 'Strong'}]), ('Supernode-0', 'Supernode-2', [{'type': 'Weak'}, {'type': 'Strong'}]), ('Supernode-1', 'Supernode-2', [{'type': 'Strong'}]), ('Supernode-1', 'Supernode-3', [{'type': 'Strong'}]), ('Supernode-3', 'Supernode-2', [{'type': 'Strong'}])]
        G = nx.DiGraph()
        for node in nodes:
            attributes = nodes[node]
            G.add_node(node, **attributes)
        for source, target, types in edges:
            G.add_edge(source, target, types=types)
        supernodes = {'Supernode-0': {'A', 'B'}, 'Supernode-1': {'C', 'D'}, 'Supernode-2': {'E', 'F'}, 'Supernode-3': {'G', 'H'}, 'Supernode-4': {'I', 'J'}, 'Supernode-5': {'K', 'L'}}
        nx.set_node_attributes(G, supernodes, 'group')
        return G