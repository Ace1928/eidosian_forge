import heapq
import inspect
import unittest
from pbr.version import VersionInfo
def _getGraph(self, resource_sets):
    """Build a graph of the resource-using nodes.

        This special cases set(['root']) to be a node with no resources and
        edges to everything.

        :return: A complete directed graph of the switching costs
            between each resource combination. Note that links from N to N are
            not included.
        """
    no_resources = frozenset()
    graph = {}
    root = set(['root'])
    for from_set in resource_sets:
        graph[from_set] = {}
        if from_set == root:
            from_resources = no_resources
        else:
            from_resources = from_set
        for to_set in resource_sets:
            if from_set is to_set:
                continue
            if to_set == root:
                continue
            else:
                to_resources = to_set
            graph[from_set][to_set] = self.cost_of_switching(from_resources, to_resources)
    return graph