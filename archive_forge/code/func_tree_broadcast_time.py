import networkx as nx
from networkx import NetworkXError
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@nx._dispatchable
def tree_broadcast_time(G, node=None):
    """Return the Broadcast Time of the tree `G`.

    The minimum broadcast time of a node is defined as the minimum amount
    of time required to complete broadcasting starting from the
    originator. The broadcast time of a graph is the maximum over
    all nodes of the minimum broadcast time from that node [1]_.
    This function returns the minimum broadcast time of `node`.
    If `node` is None the broadcast time for the graph is returned.

    Parameters
    ----------
    G : undirected graph
        The graph should be an undirected tree
    node: int, optional
        index of starting node. If `None`, the algorithm returns the broadcast
        time of the tree.

    Returns
    -------
    BT : int
        Broadcast Time of a node in a tree

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    References
    ----------
    .. [1] Harutyunyan, H. A. and Li, Z.
        "A Simple Construction of Broadcast Graphs."
        In Computing and Combinatorics. COCOON 2019
        (Ed. D. Z. Du and C. Tian.) Springer, pp. 240-253, 2019.
    """
    b_T, b_C = tree_broadcast_center(G)
    if node is not None:
        return b_T + min((nx.shortest_path_length(G, node, u) for u in b_C))
    dist_from_center = dict.fromkeys(G, len(G))
    for u in b_C:
        for v, dist in nx.shortest_path_length(G, u).items():
            if dist < dist_from_center[v]:
                dist_from_center[v] = dist
    return b_T + max(dist_from_center.values())