from itertools import combinations
import networkx as nx
from networkx.algorithms.simple_paths import is_simple_path as is_path
from networkx.utils import arbitrary_element, not_implemented_for, py_random_state
def two_neighborhood(G, v):
    """Returns the set of nodes at distance at most two from `v`.

        `G` must be a graph and `v` a node in that graph.

        The returned set includes the nodes at distance zero (that is,
        the node `v` itself), the nodes at distance one (that is, the
        out-neighbors of `v`), and the nodes at distance two.

        """
    return {x for x in G if x == v or x in G[v] or any((is_path(G, [v, z, x]) for z in G))}