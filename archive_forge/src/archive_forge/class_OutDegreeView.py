from collections.abc import Mapping, Set
import networkx as nx
class OutDegreeView(DiDegreeView):
    """A DegreeView class to report out_degree for a DiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._succ[n]
        if self._weight is None:
            return len(nbrs)
        return sum((dd.get(self._weight, 1) for dd in nbrs.values()))

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                succs = self._succ[n]
                yield (n, len(succs))
        else:
            for n in self._nodes:
                succs = self._succ[n]
                deg = sum((dd.get(weight, 1) for dd in succs.values()))
                yield (n, deg)