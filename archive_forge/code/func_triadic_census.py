from collections import defaultdict
from itertools import combinations, permutations
import networkx as nx
from networkx.utils import not_implemented_for, py_random_state
@not_implemented_for('undirected')
@nx._dispatch
def triadic_census(G, nodelist=None):
    """Determines the triadic census of a directed graph.

    The triadic census is a count of how many of the 16 possible types of
    triads are present in a directed graph. If a list of nodes is passed, then
    only those triads are taken into account which have elements of nodelist in them.

    Parameters
    ----------
    G : digraph
       A NetworkX DiGraph
    nodelist : list
        List of nodes for which you want to calculate triadic census

    Returns
    -------
    census : dict
       Dictionary with triad type as keys and number of occurrences as values.

    Examples
    --------
    >>> G = nx.DiGraph([(1, 2), (2, 3), (3, 1), (3, 4), (4, 1), (4, 2)])
    >>> triadic_census = nx.triadic_census(G)
    >>> for key, value in triadic_census.items():
    ...     print(f"{key}: {value}")
    ...
    003: 0
    012: 0
    102: 0
    021D: 0
    021U: 0
    021C: 0
    111D: 0
    111U: 0
    030T: 2
    030C: 2
    201: 0
    120D: 0
    120U: 0
    120C: 0
    210: 0
    300: 0

    Notes
    -----
    This algorithm has complexity $O(m)$ where $m$ is the number of edges in
    the graph.

    Raises
    ------
    ValueError
        If `nodelist` contains duplicate nodes or nodes not in `G`.
        If you want to ignore this you can preprocess with `set(nodelist) & G.nodes`

    See also
    --------
    triad_graph

    References
    ----------
    .. [1] Vladimir Batagelj and Andrej Mrvar, A subquadratic triad census
        algorithm for large sparse networks with small maximum degree,
        University of Ljubljana,
        http://vlado.fmf.uni-lj.si/pub/networks/doc/triads/triads.pdf

    """
    nodeset = set(G.nbunch_iter(nodelist))
    if nodelist is not None and len(nodelist) != len(nodeset):
        raise ValueError('nodelist includes duplicate nodes or nodes not in G')
    N = len(G)
    Nnot = N - len(nodeset)
    m = {n: i for i, n in enumerate(nodeset)}
    if Nnot:
        not_nodeset = G.nodes - nodeset
        m.update(((n, i + N) for i, n in enumerate(not_nodeset)))
    nbrs = {n: G.pred[n].keys() | G.succ[n].keys() for n in G}
    dbl_nbrs = {n: G.pred[n].keys() & G.succ[n].keys() for n in G}
    if Nnot:
        sgl_nbrs = {n: G.pred[n].keys() ^ G.succ[n].keys() for n in not_nodeset}
        sgl = sum((1 for n in not_nodeset for nbr in sgl_nbrs[n] if nbr not in nodeset))
        sgl_edges_outside = sgl // 2
        dbl = sum((1 for n in not_nodeset for nbr in dbl_nbrs[n] if nbr not in nodeset))
        dbl_edges_outside = dbl // 2
    census = {name: 0 for name in TRIAD_NAMES}
    for v in nodeset:
        vnbrs = nbrs[v]
        dbl_vnbrs = dbl_nbrs[v]
        if Nnot:
            sgl_unbrs_bdy = sgl_unbrs_out = dbl_unbrs_bdy = dbl_unbrs_out = 0
        for u in vnbrs:
            if m[u] <= m[v]:
                continue
            unbrs = nbrs[u]
            neighbors = (vnbrs | unbrs) - {u, v}
            for w in neighbors:
                if m[u] < m[w] or (m[v] < m[w] < m[u] and v not in nbrs[w]):
                    code = _tricode(G, v, u, w)
                    census[TRICODE_TO_NAME[code]] += 1
            if u in dbl_vnbrs:
                census['102'] += N - len(neighbors) - 2
            else:
                census['012'] += N - len(neighbors) - 2
            if Nnot and u not in nodeset:
                sgl_unbrs = sgl_nbrs[u]
                sgl_unbrs_bdy += len(sgl_unbrs & vnbrs - nodeset)
                sgl_unbrs_out += len(sgl_unbrs - vnbrs - nodeset)
                dbl_unbrs = dbl_nbrs[u]
                dbl_unbrs_bdy += len(dbl_unbrs & vnbrs - nodeset)
                dbl_unbrs_out += len(dbl_unbrs - vnbrs - nodeset)
        if Nnot:
            census['012'] += sgl_edges_outside - (sgl_unbrs_out + sgl_unbrs_bdy // 2)
            census['102'] += dbl_edges_outside - (dbl_unbrs_out + dbl_unbrs_bdy // 2)
    total_triangles = N * (N - 1) * (N - 2) // 6
    triangles_without_nodeset = Nnot * (Nnot - 1) * (Nnot - 2) // 6
    total_census = total_triangles - triangles_without_nodeset
    census['003'] = total_census - sum(census.values())
    return census