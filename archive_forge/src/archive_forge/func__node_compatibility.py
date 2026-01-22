import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@property
def _node_compatibility(self):
    if self._node_compat_ is not None:
        return self._node_compat_
    self._node_compat_ = {}
    for sgn_part_color, gn_part_color in itertools.product(range(len(self._sgn_partitions)), range(len(self._gn_partitions))):
        sgn = next(iter(self._sgn_partitions[sgn_part_color]))
        gn = next(iter(self._gn_partitions[gn_part_color]))
        if self.node_equality(self.subgraph, sgn, self.graph, gn):
            self._node_compat_[sgn_part_color] = gn_part_color
    return self._node_compat_