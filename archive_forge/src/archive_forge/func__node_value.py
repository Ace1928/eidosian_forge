import networkx as nx
def _node_value(G, node_attr):
    """Returns a function that returns a value from G.nodes[u].

    We return a function expecting a node as its sole argument. Then, in the
    simplest scenario, the returned function will return G.nodes[u][node_attr].
    However, we also handle the case when `node_attr` is None or when it is a
    function itself.

    Parameters
    ----------
    G : graph
        A NetworkX graph

    node_attr : {None, str, callable}
        Specification of how the value of the node attribute should be obtained
        from the node attribute dictionary.

    Returns
    -------
    value : function
        A function expecting a node as its sole argument. The function will
        returns a value from G.nodes[u] that depends on `edge_attr`.

    """
    if node_attr is None:

        def value(u):
            return u
    elif not callable(node_attr):

        def value(u):
            return G.nodes[u][node_attr]
    else:
        value = node_attr
    return value