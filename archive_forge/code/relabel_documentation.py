import networkx as nx
Returns a copy of the graph G with the nodes relabeled using
    consecutive integers.

    Parameters
    ----------
    G : graph
       A NetworkX graph

    first_label : int, optional (default=0)
       An integer specifying the starting offset in numbering nodes.
       The new integer labels are numbered first_label, ..., n-1+first_label.

    ordering : string
       "default" : inherit node ordering from G.nodes()
       "sorted"  : inherit node ordering from sorted(G.nodes())
       "increasing degree" : nodes are sorted by increasing degree
       "decreasing degree" : nodes are sorted by decreasing degree

    label_attribute : string, optional (default=None)
       Name of node attribute to store old label.  If None no attribute
       is created.

    Notes
    -----
    Node and edge attribute data are copied to the new (relabeled) graph.

    There is no guarantee that the relabeling of nodes to integers will
    give the same two integers for two (even identical graphs).
    Use the `ordering` argument to try to preserve the order.

    See Also
    --------
    relabel_nodes
    