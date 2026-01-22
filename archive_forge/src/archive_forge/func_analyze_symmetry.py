import itertools
from collections import Counter, defaultdict
from functools import reduce, wraps
def analyze_symmetry(self, graph, node_partitions, edge_colors):
    """
        Find a minimal set of permutations and corresponding co-sets that
        describe the symmetry of `graph`, given the node and edge equalities
        given by `node_partitions` and `edge_colors`, respectively.

        Parameters
        ----------
        graph : networkx.Graph
            The graph whose symmetry should be analyzed.
        node_partitions : list of sets
            A list of sets containing node keys. Node keys in the same set
            are considered equivalent. Every node key in `graph` should be in
            exactly one of the sets. If all nodes are equivalent, this should
            be ``[set(graph.nodes)]``.
        edge_colors : dict mapping edges to their colors
            A dict mapping every edge in `graph` to its corresponding color.
            Edges with the same color are considered equivalent. If all edges
            are equivalent, this should be ``{e: 0 for e in graph.edges}``.


        Returns
        -------
        set[frozenset]
            The found permutations. This is a set of frozensets of pairs of node
            keys which can be exchanged without changing :attr:`subgraph`.
        dict[collections.abc.Hashable, set[collections.abc.Hashable]]
            The found co-sets. The co-sets is a dictionary of
            ``{node key: set of node keys}``.
            Every key-value pair describes which ``values`` can be interchanged
            without changing nodes less than ``key``.
        """
    if self._symmetry_cache is not None:
        key = hash((tuple(graph.nodes), tuple(graph.edges), tuple(map(tuple, node_partitions)), tuple(edge_colors.items())))
        if key in self._symmetry_cache:
            return self._symmetry_cache[key]
    node_partitions = list(self._refine_node_partitions(graph, node_partitions, edge_colors))
    assert len(node_partitions) == 1
    node_partitions = node_partitions[0]
    permutations, cosets = self._process_ordered_pair_partitions(graph, node_partitions, node_partitions, edge_colors)
    if self._symmetry_cache is not None:
        self._symmetry_cache[key] = (permutations, cosets)
    return (permutations, cosets)