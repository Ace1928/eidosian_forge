from collections.abc import Mapping, Set
import networkx as nx
class InEdgeView(OutEdgeView):
    """A EdgeView class for inward edges of a DiGraph"""
    __slots__ = ()

    def __setstate__(self, state):
        self._graph = state['_graph']
        self._adjdict = state['_adjdict']
        self._nodes_nbrs = self._adjdict.items
    dataview = InEdgeDataView

    def __init__(self, G):
        self._graph = G
        self._adjdict = G._pred if hasattr(G, 'pred') else G._adj
        self._nodes_nbrs = self._adjdict.items

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr in nbrs:
                yield (nbr, n)

    def __contains__(self, e):
        try:
            u, v = e
            return u in self._adjdict[v]
        except KeyError:
            return False

    def __getitem__(self, e):
        if isinstance(e, slice):
            raise nx.NetworkXError(f'{type(self).__name__} does not support slicing, try list(G.in_edges)[{e.start}:{e.stop}:{e.step}]')
        u, v = e
        return self._adjdict[v][u]