from collections.abc import Mapping, Set
import networkx as nx
class InEdgeDataView(OutEdgeDataView):
    """An EdgeDataView class for outward edges of DiGraph; See EdgeDataView"""
    __slots__ = ()

    def __iter__(self):
        return (self._report(nbr, n, dd) for n, nbrs in self._nodes_nbrs() for nbr, dd in nbrs.items())

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and v not in self._nbunch:
            return False
        try:
            ddict = self._adjdict[v][u]
        except KeyError:
            return False
        return e == self._report(u, v, ddict)