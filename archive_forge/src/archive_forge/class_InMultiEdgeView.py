from collections.abc import Mapping, Set
import networkx as nx
class InMultiEdgeView(OutMultiEdgeView):
    """A EdgeView class for inward edges of a MultiDiGraph"""
    __slots__ = ()

    def __setstate__(self, state):
        self._graph = state['_graph']
        self._adjdict = state['_adjdict']
        self._nodes_nbrs = self._adjdict.items
    dataview = InMultiEdgeDataView

    def __init__(self, G):
        self._graph = G
        self._adjdict = G._pred if hasattr(G, 'pred') else G._adj
        self._nodes_nbrs = self._adjdict.items

    def __iter__(self):
        for n, nbrs in self._nodes_nbrs():
            for nbr, kdict in nbrs.items():
                for key in kdict:
                    yield (nbr, n, key)

    def __contains__(self, e):
        N = len(e)
        if N == 3:
            u, v, k = e
        elif N == 2:
            u, v = e
            k = 0
        else:
            raise ValueError('MultiEdge must have length 2 or 3')
        try:
            return k in self._adjdict[v][u]
        except KeyError:
            return False

    def __getitem__(self, e):
        if isinstance(e, slice):
            raise nx.NetworkXError(f'{type(self).__name__} does not support slicing, try list(G.in_edges)[{e.start}:{e.stop}:{e.step}]')
        u, v, k = e
        return self._adjdict[v][u][k]