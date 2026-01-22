import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
@property
def _sgn_partitions(self):
    if self._sgn_partitions_ is None:

        def nodematch(node1, node2):
            return self.node_equality(self.subgraph, node1, self.subgraph, node2)
        self._sgn_partitions_ = make_partitions(self.subgraph.nodes, nodematch)
    return self._sgn_partitions_