from itertools import chain, repeat
import networkx as nx
@nx._dispatch(graphs='[graphs]', preserve_all_attrs=True)
def union_all(graphs, rename=()):
    """Returns the union of all graphs.

    The graphs must be disjoint, otherwise an exception is raised.

    Parameters
    ----------
    graphs : iterable
       Iterable of NetworkX graphs

    rename : iterable , optional
       Node names of graphs can be changed by specifying the tuple
       rename=('G-','H-') (for example).  Node "u" in G is then renamed
       "G-u" and "v" in H is renamed "H-v". Infinite generators (like itertools.count)
       are also supported.

    Returns
    -------
    U : a graph with the same type as the first graph in list

    Raises
    ------
    ValueError
       If `graphs` is an empty list.

    NetworkXError
        In case of mixed type graphs, like MultiGraph and Graph, or directed and undirected graphs.

    Notes
    -----
    For operating on mixed type graphs, they should be converted to the same type.
    >>> G = nx.Graph()
    >>> H = nx.DiGraph()
    >>> GH = union_all([nx.DiGraph(G), H])

    To force a disjoint union with node relabeling, use
    disjoint_union_all(G,H) or convert_node_labels_to integers().

    Graph, edge, and node attributes are propagated to the union graph.
    If a graph attribute is present in multiple graphs, then the value
    from the last graph in the list with that attribute is used.

    Examples
    --------
    >>> G1 = nx.Graph([(1, 2), (2, 3)])
    >>> G2 = nx.Graph([(4, 5), (5, 6)])
    >>> result_graph = nx.union_all([G1, G2])
    >>> result_graph.nodes()
    NodeView((1, 2, 3, 4, 5, 6))
    >>> result_graph.edges()
    EdgeView([(1, 2), (2, 3), (4, 5), (5, 6)])

    See Also
    --------
    union
    disjoint_union_all
    """
    R = None
    seen_nodes = set()

    def add_prefix(graph, prefix):
        if prefix is None:
            return graph

        def label(x):
            return f'{prefix}{x}'
        return nx.relabel_nodes(graph, label)
    rename = chain(rename, repeat(None))
    graphs = (add_prefix(G, name) for G, name in zip(graphs, rename))
    for i, G in enumerate(graphs):
        G_nodes_set = set(G.nodes)
        if i == 0:
            R = G.__class__()
        elif G.is_directed() != R.is_directed():
            raise nx.NetworkXError('All graphs must be directed or undirected.')
        elif G.is_multigraph() != R.is_multigraph():
            raise nx.NetworkXError('All graphs must be graphs or multigraphs.')
        elif not seen_nodes.isdisjoint(G_nodes_set):
            raise nx.NetworkXError('The node sets of the graphs are not disjoint.', 'Use appropriate rename=(G1prefix,G2prefix,...,GNprefix)or use disjoint_union(G1,G2,...,GN).')
        seen_nodes |= G_nodes_set
        R.graph.update(G.graph)
        R.add_nodes_from(G.nodes(data=True))
        R.add_edges_from(G.edges(keys=True, data=True) if G.is_multigraph() else G.edges(data=True))
    if R is None:
        raise ValueError('cannot apply union_all to an empty list')
    return R