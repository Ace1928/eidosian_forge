from collections import defaultdict
from collections.abc import Mapping, Set
from itertools import combinations_with_replacement
import networkx as nx
from networkx.utils import UnionFind, arbitrary_element, not_implemented_for
@not_implemented_for('undirected')
@nx._dispatch
def tree_all_pairs_lowest_common_ancestor(G, root=None, pairs=None):
    """Yield the lowest common ancestor for sets of pairs in a tree.

    Parameters
    ----------
    G : NetworkX directed graph (must be a tree)

    root : node, optional (default: None)
        The root of the subtree to operate on.
        If None, assume the entire graph has exactly one source and use that.

    pairs : iterable or iterator of pairs of nodes, optional (default: None)
        The pairs of interest. If None, Defaults to all pairs of nodes
        under `root` that have a lowest common ancestor.

    Returns
    -------
    lcas : generator of tuples `((u, v), lca)` where `u` and `v` are nodes
        in `pairs` and `lca` is their lowest common ancestor.

    Examples
    --------
    >>> import pprint
    >>> G = nx.DiGraph([(1, 3), (2, 4), (1, 2)])
    >>> pprint.pprint(dict(nx.tree_all_pairs_lowest_common_ancestor(G)))
    {(1, 1): 1,
     (2, 1): 1,
     (2, 2): 2,
     (3, 1): 1,
     (3, 2): 1,
     (3, 3): 3,
     (3, 4): 1,
     (4, 1): 1,
     (4, 2): 2,
     (4, 4): 4}

    We can also use `pairs` argument to specify the pairs of nodes for which we
    want to compute lowest common ancestors. Here is an example:

    >>> dict(nx.tree_all_pairs_lowest_common_ancestor(G, pairs=[(1, 4), (2, 3)]))
    {(2, 3): 1, (1, 4): 1}

    Notes
    -----
    Only defined on non-null trees represented with directed edges from
    parents to children. Uses Tarjan's off-line lowest-common-ancestors
    algorithm. Runs in time $O(4 \\times (V + E + P))$ time, where 4 is the largest
    value of the inverse Ackermann function likely to ever come up in actual
    use, and $P$ is the number of pairs requested (or $V^2$ if all are needed).

    Tarjan, R. E. (1979), "Applications of path compression on balanced trees",
    Journal of the ACM 26 (4): 690-715, doi:10.1145/322154.322161.

    See Also
    --------
    all_pairs_lowest_common_ancestor: similar routine for general DAGs
    lowest_common_ancestor: just a single pair for general DAGs
    """
    if len(G) == 0:
        raise nx.NetworkXPointlessConcept('LCA meaningless on null graphs.')
    if pairs is not None:
        pair_dict = defaultdict(set)
        if not isinstance(pairs, (Mapping, Set)):
            pairs = set(pairs)
        for u, v in pairs:
            for n in (u, v):
                if n not in G:
                    msg = f'The node {str(n)} is not in the digraph.'
                    raise nx.NodeNotFound(msg)
            pair_dict[u].add(v)
            pair_dict[v].add(u)
    if root is None:
        for n, deg in G.in_degree:
            if deg == 0:
                if root is not None:
                    msg = 'No root specified and tree has multiple sources.'
                    raise nx.NetworkXError(msg)
                root = n
            elif deg > 1 and len(G.pred[n]) > 1:
                msg = 'Tree LCA only defined on trees; use DAG routine.'
                raise nx.NetworkXError(msg)
    if root is None:
        raise nx.NetworkXError('Graph contains a cycle.')
    uf = UnionFind()
    ancestors = {}
    for node in G:
        ancestors[node] = uf[node]
    colors = defaultdict(bool)
    for node in nx.dfs_postorder_nodes(G, root):
        colors[node] = True
        for v in pair_dict[node] if pairs is not None else G:
            if colors[v]:
                if pairs is not None and (node, v) in pairs:
                    yield ((node, v), ancestors[uf[v]])
                if pairs is None or (v, node) in pairs:
                    yield ((v, node), ancestors[uf[v]])
        if node != root:
            parent = arbitrary_element(G.pred[node])
            uf.union(parent, node)
            ancestors[uf[parent]] = parent