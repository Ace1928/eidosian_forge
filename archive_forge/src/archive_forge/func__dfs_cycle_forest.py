import networkx as nx
from networkx.utils import not_implemented_for
def _dfs_cycle_forest(G, root=None):
    """Builds a directed graph composed of cycles from the given graph.

        `G` is an undirected simple graph. `root` is a node in the graph
        from which the depth-first search is started.

        This function returns both the depth-first search cycle graph
        (as a :class:`~networkx.DiGraph`) and the list of nodes in
        depth-first preorder. The depth-first search cycle graph is a
        directed graph whose edges are the edges of `G` oriented toward
        the root if the edge is a tree edge and away from the root if
        the edge is a non-tree edge. If `root` is not specified, this
        performs a depth-first search on each connected component of `G`
        and returns a directed forest instead.

        If `root` is not in the graph, this raises :exc:`KeyError`.

        """
    H = nx.DiGraph()
    nodes = []
    for u, v, d in nx.dfs_labeled_edges(G, source=root):
        if d == 'forward':
            if u == v:
                H.add_node(v, parent=None)
                nodes.append(v)
            else:
                H.add_node(v, parent=u)
                H.add_edge(v, u, nontree=False)
                nodes.append(v)
        elif d == 'nontree' and v not in H[u]:
            H.add_edge(v, u, nontree=True)
        else:
            pass
    return (H, nodes)