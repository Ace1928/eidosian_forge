import itertools
import pytest
import networkx as nx
def two_node_graph():
    graph = nx.Graph()
    graph.add_nodes_from([1, 2])
    graph.add_edges_from([(1, 2)])
    return graph