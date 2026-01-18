import networkx as nx

    Returns True if `G` is a tree.

    A tree is a connected graph with no undirected cycles.

    For directed graphs, `G` is a tree if the underlying graph is a tree. The
    underlying graph is obtained by treating each directed edge as a single
    undirected edge in a multigraph.

    Parameters
    ----------
    G : graph
        The graph to test.

    Returns
    -------
    b : bool
        A boolean that is True if `G` is a tree.

    Raises
    ------
    NetworkXPointlessConcept
        If `G` is empty.

    Examples
    --------
    >>> G = nx.Graph()
    >>> G.add_edges_from([(1, 2), (1, 3), (2, 4), (2, 5)])
    >>> nx.is_tree(G)  # n-1 edges
    True
    >>> G.add_edge(3, 4)
    >>> nx.is_tree(G)  # n edges
    False

    Notes
    -----
    In another convention, a directed tree is known as a *polytree* and then
    *tree* corresponds to an *arborescence*.

    See Also
    --------
    is_arborescence

    