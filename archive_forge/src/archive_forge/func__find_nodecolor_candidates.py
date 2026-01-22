import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def _find_nodecolor_candidates(self):
    """
        Per node in subgraph find all nodes in graph that have the same color.
        """
    candidates = defaultdict(set)
    for sgn in self.subgraph.nodes:
        sgn_color = self._sgn_colors[sgn]
        if sgn_color in self._node_compatibility:
            gn_color = self._node_compatibility[sgn_color]
            candidates[sgn].add(frozenset(self._gn_partitions[gn_color]))
        else:
            candidates[sgn].add(frozenset())
    candidates = dict(candidates)
    for sgn, options in candidates.items():
        candidates[sgn] = frozenset(options)
    return candidates