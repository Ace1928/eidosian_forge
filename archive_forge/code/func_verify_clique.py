import pytest
import networkx as nx
def verify_clique(graph, clique, reported_clique_weight, expected_clique_weight, weight_accessor):
    for node1 in clique:
        for node2 in clique:
            if node1 == node2:
                continue
            if not graph.has_edge(node1, node2):
                return False
    if weight_accessor is None:
        clique_weight = len(clique)
    else:
        clique_weight = sum((graph.nodes[v]['weight'] for v in clique))
    if clique_weight != expected_clique_weight:
        return False
    if clique_weight != reported_clique_weight:
        return False
    return True