import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def _resource_graph(resource_sets):
    """Convert an iterable of resource_sets into a graph.

    Each resource_set in the iterable is treated as a node, and each resource
    in that resource_set is used as an edge to other nodes.
    """
    nodes = {}
    edges = {}
    for resource_set in resource_sets:
        node = frozenset(resource_set)
        nodes[node] = set()
        for resource in resource_set:
            edges.setdefault(resource, []).append(node)
    for node, connected in nodes.items():
        for resource in node:
            connected.update(edges[resource])
        connected.discard(node)
    return nodes