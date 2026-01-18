from itertools import chain
import networkx as nx
@nx._dispatch(graphs=None)
def tree_graph(data, ident='id', children='children'):
    """Returns graph from tree data format.

    Parameters
    ----------
    data : dict
        Tree formatted graph data

    ident : string
        Attribute name for storing NetworkX-internal graph data. `ident` must
        have a different value than `children`. The default is 'id'.

    children : string
        Attribute name for storing NetworkX-internal graph data. `children`
        must have a different value than `ident`. The default is 'children'.

    Returns
    -------
    G : NetworkX DiGraph

    Examples
    --------
    >>> from networkx.readwrite import json_graph
    >>> G = nx.DiGraph([(1, 2)])
    >>> data = json_graph.tree_data(G, root=1)
    >>> H = json_graph.tree_graph(data)

    See Also
    --------
    tree_data, node_link_data, adjacency_data
    """
    graph = nx.DiGraph()

    def add_children(parent, children_):
        for data in children_:
            child = data[ident]
            graph.add_edge(parent, child)
            grandchildren = data.get(children, [])
            if grandchildren:
                add_children(child, grandchildren)
            nodedata = {str(k): v for k, v in data.items() if k != ident and k != children}
            graph.add_node(child, **nodedata)
    root = data[ident]
    children_ = data.get(children, [])
    nodedata = {str(k): v for k, v in data.items() if k != ident and k != children}
    graph.add_node(root, **nodedata)
    add_children(root, children_)
    return graph