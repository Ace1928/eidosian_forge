import itertools
import numbers
import networkx as nx
from networkx.classes import Graph
from networkx.exception import NetworkXError
from networkx.utils import nodes_or_number, pairwise
Returns the complete multipartite graph with the specified subset sizes.

    Parameters
    ----------
    subset_sizes : tuple of integers or tuple of node iterables
       The arguments can either all be integer number of nodes or they
       can all be iterables of nodes. If integers, they represent the
       number of nodes in each subset of the multipartite graph.
       If iterables, each is used to create the nodes for that subset.
       The length of subset_sizes is the number of subsets.

    Returns
    -------
    G : NetworkX Graph
       Returns the complete multipartite graph with the specified subsets.

       For each node, the node attribute 'subset' is an integer
       indicating which subset contains the node.

    Examples
    --------
    Creating a complete tripartite graph, with subsets of one, two, and three
    nodes, respectively.

        >>> G = nx.complete_multipartite_graph(1, 2, 3)
        >>> [G.nodes[u]["subset"] for u in G]
        [0, 1, 1, 2, 2, 2]
        >>> list(G.edges(0))
        [(0, 1), (0, 2), (0, 3), (0, 4), (0, 5)]
        >>> list(G.edges(2))
        [(2, 0), (2, 3), (2, 4), (2, 5)]
        >>> list(G.edges(4))
        [(4, 0), (4, 1), (4, 2)]

        >>> G = nx.complete_multipartite_graph("a", "bc", "def")
        >>> [G.nodes[u]["subset"] for u in sorted(G)]
        [0, 1, 1, 2, 2, 2]

    Notes
    -----
    This function generalizes several other graph builder functions.

    - If no subset sizes are given, this returns the null graph.
    - If a single subset size `n` is given, this returns the empty graph on
      `n` nodes.
    - If two subset sizes `m` and `n` are given, this returns the complete
      bipartite graph on `m + n` nodes.
    - If subset sizes `1` and `n` are given, this returns the star graph on
      `n + 1` nodes.

    See also
    --------
    complete_bipartite_graph
    