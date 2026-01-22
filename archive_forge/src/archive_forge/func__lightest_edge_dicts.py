import math
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
def _lightest_edge_dicts(residual_graph, clustering, node):
    """Find the lightest edge to each cluster.

    Searches for the minimum-weight edge to each cluster adjacent to
    the given node.

    Parameters
    ----------
    residual_graph : NetworkX graph
        The residual graph used by the Baswana-Sen algorithm.

    clustering : dictionary
        The current clustering of the nodes.

    node : node
        The node from which the search originates.

    Returns
    -------
    lightest_edge_neighbor, lightest_edge_weight : dictionary, dictionary
        lightest_edge_neighbor is a dictionary that maps a center C to
        a node v in the corresponding cluster such that the edge from
        the given node to v is the lightest edge from the given node to
        any node in cluster. lightest_edge_weight maps a center C to the
        weight of the aforementioned edge.

    Notes
    -----
    If a cluster has no node that is adjacent to the given node in the
    residual graph then the center of the cluster is not a key in the
    returned dictionaries.
    """
    lightest_edge_neighbor = {}
    lightest_edge_weight = {}
    for neighbor in residual_graph.adj[node]:
        neighbor_center = clustering[neighbor]
        weight = residual_graph[node][neighbor]['weight']
        if neighbor_center not in lightest_edge_weight or weight < lightest_edge_weight[neighbor_center]:
            lightest_edge_neighbor[neighbor_center] = neighbor
            lightest_edge_weight[neighbor_center] = weight
    return (lightest_edge_neighbor, lightest_edge_weight)