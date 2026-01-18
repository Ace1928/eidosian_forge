import itertools
import pytest
import networkx as nx
def verify_coloring(graph, coloring):
    for node in graph.nodes():
        if node not in coloring:
            return False
        color = coloring[node]
        for neighbor in graph.neighbors(node):
            if coloring[neighbor] == color:
                return False
    return True