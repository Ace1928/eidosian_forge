import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@property
def _sge_partitions(self):
    if self._sge_partitions_ is None:

        def edgematch(edge1, edge2):
            return self.edge_equality(self.subgraph, edge1, self.subgraph, edge2)
        self._sge_partitions_ = make_partitions(self.subgraph.edges, edgematch)
    return self._sge_partitions_