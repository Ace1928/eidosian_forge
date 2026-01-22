import pytest
import networkx as nx
class TestSNAPUndirectedMulti(AbstractSNAP):

    def build_original_graph(self):
        nodes = {'A': {'color': 'Red'}, 'B': {'color': 'Red'}, 'C': {'color': 'Red'}, 'D': {'color': 'Blue'}, 'E': {'color': 'Blue'}, 'F': {'color': 'Blue'}, 'G': {'color': 'Yellow'}, 'H': {'color': 'Yellow'}, 'I': {'color': 'Yellow'}}
        edges = [('A', 'D', ['Weak', 'Strong']), ('B', 'E', ['Weak', 'Strong']), ('D', 'I', ['Strong']), ('E', 'H', ['Strong']), ('F', 'G', ['Weak']), ('I', 'G', ['Weak', 'Strong']), ('I', 'H', ['Weak', 'Strong']), ('G', 'H', ['Weak', 'Strong'])]
        G = nx.MultiGraph()
        for node in nodes:
            attributes = nodes[node]
            G.add_node(node, **attributes)
        for source, target, types in edges:
            for type in types:
                G.add_edge(source, target, type=type)
        return G

    def build_summary_graph(self):
        nodes = {'Supernode-0': {'color': 'Red'}, 'Supernode-1': {'color': 'Blue'}, 'Supernode-2': {'color': 'Yellow'}, 'Supernode-3': {'color': 'Blue'}, 'Supernode-4': {'color': 'Yellow'}, 'Supernode-5': {'color': 'Red'}}
        edges = [('Supernode-1', 'Supernode-2', [{'type': 'Weak'}]), ('Supernode-2', 'Supernode-4', [{'type': 'Weak'}, {'type': 'Strong'}]), ('Supernode-3', 'Supernode-4', [{'type': 'Strong'}]), ('Supernode-3', 'Supernode-5', [{'type': 'Weak'}, {'type': 'Strong'}]), ('Supernode-4', 'Supernode-4', [{'type': 'Weak'}, {'type': 'Strong'}])]
        G = nx.MultiGraph()
        for node in nodes:
            attributes = nodes[node]
            G.add_node(node, **attributes)
        for source, target, types in edges:
            for type in types:
                G.add_edge(source, target, type=type)
        supernodes = {'Supernode-0': {'A', 'B'}, 'Supernode-1': {'C', 'D'}, 'Supernode-2': {'E', 'F'}, 'Supernode-3': {'G', 'H'}, 'Supernode-4': {'I', 'J'}, 'Supernode-5': {'K', 'L'}}
        nx.set_node_attributes(G, supernodes, 'group')
        return G