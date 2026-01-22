import warnings
import networkx as nx
@nx._dispatch(edge_attrs='weight')
def all_pairs_all_shortest_paths(G, weight=None, method='dijkstra'):
    """Compute all shortest paths between all nodes.

    Parameters
    ----------
    G : NetworkX graph

    weight : None, string or function, optional (default = None)
        If None, every edge has weight/distance/cost 1.
        If a string, use this edge attribute as the edge weight.
        Any edge attribute not present defaults to 1.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly
        three positional arguments: the two endpoints of an edge and
        the dictionary of edge attributes for that edge.
        The function must return a number.

    method : string, optional (default = 'dijkstra')
       The algorithm to use to compute the path lengths.
       Supported options: 'dijkstra', 'bellman-ford'.
       Other inputs produce a ValueError.
       If `weight` is None, unweighted graph methods are used, and this
       suggestion is ignored.

    Returns
    -------
    paths : generator of dictionary
        Dictionary of arrays, keyed by source and target, of all shortest paths.

    Raises
    ------
    ValueError
        If `method` is not among the supported options.

    Examples
    --------
    >>> G = nx.cycle_graph(4)
    >>> dict(nx.all_pairs_all_shortest_paths(G))[0][2]
    [[0, 1, 2], [0, 3, 2]]
    >>> dict(nx.all_pairs_all_shortest_paths(G))[0][3]
    [[0, 3]]

    Notes
    -----
    There may be multiple shortest paths with equal lengths. Unlike
    all_pairs_shortest_path, this method returns all shortest paths.

    See Also
    --------
    all_pairs_shortest_path
    single_source_all_shortest_paths
    """
    for n in G:
        yield (n, dict(single_source_all_shortest_paths(G, n, weight=weight, method=method)))