import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def _largest_common_subgraph(self, candidates, constraints, to_be_mapped=None):
    """
        Find all largest common subgraphs honoring constraints.
        """
    if to_be_mapped is None:
        to_be_mapped = {frozenset(self.subgraph.nodes)}
    current_size = len(next(iter(to_be_mapped), []))
    found_iso = False
    if current_size <= len(self.graph):
        for nodes in sorted(to_be_mapped, key=sorted):
            next_sgn = min(nodes, key=lambda n: min(candidates[n], key=len))
            isomorphs = self._map_nodes(next_sgn, candidates, constraints, to_be_mapped=nodes)
            try:
                item = next(isomorphs)
            except StopIteration:
                pass
            else:
                yield item
                yield from isomorphs
                found_iso = True
    if found_iso or current_size == 1:
        return
    left_to_be_mapped = set()
    for nodes in to_be_mapped:
        for sgn in nodes:
            new_nodes = self._remove_node(sgn, nodes, constraints)
            left_to_be_mapped.add(new_nodes)
    yield from self._largest_common_subgraph(candidates, constraints, to_be_mapped=left_to_be_mapped)