from collections.abc import Mapping, Set
import networkx as nx
class MultiEdgeDataView(OutMultiEdgeDataView):
    """An EdgeDataView class for edges of MultiGraph; See EdgeDataView"""
    __slots__ = ()

    def __iter__(self):
        seen = {}
        for n, nbrs in self._nodes_nbrs():
            for nbr, kd in nbrs.items():
                if nbr not in seen:
                    for k, dd in kd.items():
                        yield self._report(n, nbr, k, dd)
            seen[n] = 1
        del seen

    def __contains__(self, e):
        u, v = e[:2]
        if self._nbunch is not None and u not in self._nbunch and (v not in self._nbunch):
            return False
        try:
            kdict = self._adjdict[u][v]
        except KeyError:
            try:
                kdict = self._adjdict[v][u]
            except KeyError:
                return False
        if self.keys is True:
            k = e[2]
            try:
                dd = kdict[k]
            except KeyError:
                return False
            return e == self._report(u, v, k, dd)
        return any((e == self._report(u, v, k, dd) for k, dd in kdict.items()))