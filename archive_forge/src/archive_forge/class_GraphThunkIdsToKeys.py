import time
from . import debug, errors, osutils, revision, trace
class GraphThunkIdsToKeys:
    """Forwards calls about 'ids' to be about keys internally."""

    def __init__(self, graph):
        self._graph = graph

    def topo_sort(self):
        return [r for r, in self._graph.topo_sort()]

    def heads(self, ids):
        """See Graph.heads()"""
        as_keys = [(i,) for i in ids]
        head_keys = self._graph.heads(as_keys)
        return {h[0] for h in head_keys}

    def merge_sort(self, tip_revision):
        nodes = self._graph.merge_sort((tip_revision,))
        for node in nodes:
            node.key = node.key[0]
        return nodes

    def add_node(self, revision, parents):
        self._graph.add_node((revision,), [(p,) for p in parents])