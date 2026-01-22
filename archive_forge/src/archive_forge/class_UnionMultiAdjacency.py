from collections.abc import Mapping
class UnionMultiAdjacency(UnionAdjacency):
    """A read-only union of two dict MultiAdjacencies.

    The two input dict-of-dict-of-dict-of-dicts represent the union of
    `G.succ` and `G.pred` for MultiDiGraphs. Return values are UnionAdjacency.
    The inner level of dict is read-write. But the outer levels are read-only.

    See Also
    ========
    UnionAtlas:  View into dict-of-dict
    UnionMultiInner:  View into dict-of-dict-of-dict
    """
    __slots__ = ()

    def __getitem__(self, node):
        return UnionMultiInner(self._succ[node], self._pred[node])