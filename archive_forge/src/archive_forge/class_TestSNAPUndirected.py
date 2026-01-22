import pytest
import networkx as nx
class TestSNAPUndirected(AbstractSNAP):

    def build_original_graph(self):
        nodes = {'A': {'color': 'Red'}, 'B': {'color': 'Red'}, 'C': {'color': 'Red'}, 'D': {'color': 'Red'}, 'E': {'color': 'Blue'}, 'F': {'color': 'Blue'}, 'G': {'color': 'Blue'}, 'H': {'color': 'Blue'}, 'I': {'color': 'Yellow'}, 'J': {'color': 'Yellow'}, 'K': {'color': 'Yellow'}, 'L': {'color': 'Yellow'}}
        edges = [('A', 'B', 'Strong'), ('A', 'C', 'Weak'), ('A', 'E', 'Strong'), ('A', 'I', 'Weak'), ('B', 'D', 'Weak'), ('B', 'J', 'Weak'), ('B', 'F', 'Strong'), ('C', 'G', 'Weak'), ('D', 'H', 'Weak'), ('I', 'J', 'Strong'), ('J', 'K', 'Strong'), ('I', 'L', 'Strong')]
        G = nx.Graph()
        for node in nodes:
            attributes = nodes[node]
            G.add_node(node, **attributes)
        for source, target, type in edges:
            G.add_edge(source, target, type=type)
        return G

    def build_summary_graph(self):
        nodes = {'Supernode-0': {'color': 'Red'}, 'Supernode-1': {'color': 'Red'}, 'Supernode-2': {'color': 'Blue'}, 'Supernode-3': {'color': 'Blue'}, 'Supernode-4': {'color': 'Yellow'}, 'Supernode-5': {'color': 'Yellow'}}
        edges = [('Supernode-0', 'Supernode-0', 'Strong'), ('Supernode-0', 'Supernode-1', 'Weak'), ('Supernode-0', 'Supernode-2', 'Strong'), ('Supernode-0', 'Supernode-4', 'Weak'), ('Supernode-1', 'Supernode-3', 'Weak'), ('Supernode-4', 'Supernode-4', 'Strong'), ('Supernode-4', 'Supernode-5', 'Strong')]
        G = nx.Graph()
        for node in nodes:
            attributes = nodes[node]
            G.add_node(node, **attributes)
        for source, target, type in edges:
            G.add_edge(source, target, types=[{'type': type}])
        supernodes = {'Supernode-0': {'A', 'B'}, 'Supernode-1': {'C', 'D'}, 'Supernode-2': {'E', 'F'}, 'Supernode-3': {'G', 'H'}, 'Supernode-4': {'I', 'J'}, 'Supernode-5': {'K', 'L'}}
        nx.set_node_attributes(G, supernodes, 'group')
        return G