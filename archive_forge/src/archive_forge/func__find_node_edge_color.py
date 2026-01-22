import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@staticmethod
def _find_node_edge_color(graph, node_colors, edge_colors):
    """
        For every node in graph, come up with a color that combines 1) the
        color of the node, and 2) the number of edges of a color to each type
        of node.
        """
    counts = defaultdict(lambda: defaultdict(int))
    for node1, node2 in graph.edges:
        if (node1, node2) in edge_colors:
            ecolor = edge_colors[node1, node2]
        else:
            ecolor = edge_colors[node2, node1]
        counts[node1][ecolor, node_colors[node2]] += 1
        counts[node2][ecolor, node_colors[node1]] += 1
    node_edge_colors = {}
    for node in graph.nodes:
        node_edge_colors[node] = (node_colors[node], set(counts[node].items()))
    return node_edge_colors