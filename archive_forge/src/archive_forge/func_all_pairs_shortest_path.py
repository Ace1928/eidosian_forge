import warnings
import networkx as nx
@nx._dispatch
def all_pairs_shortest_path(G, cutoff=None):
    """Compute shortest paths between all nodes.

    Parameters
    ----------
    G : NetworkX graph

    cutoff : integer, optional
        Depth at which to stop the search. Only paths of length at most
        `cutoff` are returned.

    Returns
    -------
    paths : iterator
        Dictionary, keyed by source and target, of shortest paths.

    Examples
    --------
    >>> G = nx.path_graph(5)
    >>> path = dict(nx.all_pairs_shortest_path(G))
    >>> print(path[0][4])
    [0, 1, 2, 3, 4]

    Notes
    -----
    There may be multiple shortest paths with the same length between
    two nodes. For each pair, this function returns only one of those paths.

    See Also
    --------
    floyd_warshall
    all_pairs_all_shortest_paths

    """
    for n in G:
        yield (n, single_source_shortest_path(G, n, cutoff=cutoff))