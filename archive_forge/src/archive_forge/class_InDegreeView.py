from collections.abc import Mapping, Set
import networkx as nx
class InDegreeView(DiDegreeView):
    """A DegreeView class to report in_degree for a DiGraph; See DegreeView"""

    def __getitem__(self, n):
        weight = self._weight
        nbrs = self._pred[n]
        if weight is None:
            return len(nbrs)
        return sum((dd.get(weight, 1) for dd in nbrs.values()))

    def __iter__(self):
        weight = self._weight
        if weight is None:
            for n in self._nodes:
                preds = self._pred[n]
                yield (n, len(preds))
        else:
            for n in self._nodes:
                preds = self._pred[n]
                deg = sum((dd.get(weight, 1) for dd in preds.values()))
                yield (n, deg)