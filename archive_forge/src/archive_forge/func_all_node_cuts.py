import copy
from collections import defaultdict
from itertools import combinations
from operator import itemgetter
import networkx as nx
from networkx.algorithms.flow import (
from .utils import build_auxiliary_node_connectivity
@nx._dispatch
def all_node_cuts(G, k=None, flow_func=None):
    """Returns all minimum k cutsets of an undirected graph G.

    This implementation is based on Kanevsky's algorithm [1]_ for finding all
    minimum-size node cut-sets of an undirected graph G; ie the set (or sets)
    of nodes of cardinality equal to the node connectivity of G. Thus if
    removed, would break G into two or more connected components.

    Parameters
    ----------
    G : NetworkX graph
        Undirected graph

    k : Integer
        Node connectivity of the input graph. If k is None, then it is
        computed. Default value: None.

    flow_func : function
        Function to perform the underlying flow computations. Default value is
        :func:`~networkx.algorithms.flow.edmonds_karp`. This function performs
        better in sparse graphs with right tailed degree distributions.
        :func:`~networkx.algorithms.flow.shortest_augmenting_path` will
        perform better in denser graphs.


    Returns
    -------
    cuts : a generator of node cutsets
        Each node cutset has cardinality equal to the node connectivity of
        the input graph.

    Examples
    --------
    >>> # A two-dimensional grid graph has 4 cutsets of cardinality 2
    >>> G = nx.grid_2d_graph(5, 5)
    >>> cutsets = list(nx.all_node_cuts(G))
    >>> len(cutsets)
    4
    >>> all(2 == len(cutset) for cutset in cutsets)
    True
    >>> nx.node_connectivity(G)
    2

    Notes
    -----
    This implementation is based on the sequential algorithm for finding all
    minimum-size separating vertex sets in a graph [1]_. The main idea is to
    compute minimum cuts using local maximum flow computations among a set
    of nodes of highest degree and all other non-adjacent nodes in the Graph.
    Once we find a minimum cut, we add an edge between the high degree
    node and the target node of the local maximum flow computation to make
    sure that we will not find that minimum cut again.

    See also
    --------
    node_connectivity
    edmonds_karp
    shortest_augmenting_path

    References
    ----------
    .. [1]  Kanevsky, A. (1993). Finding all minimum-size separating vertex
            sets in a graph. Networks 23(6), 533--541.
            http://onlinelibrary.wiley.com/doi/10.1002/net.3230230604/abstract

    """
    if not nx.is_connected(G):
        raise nx.NetworkXError('Input graph is disconnected.')
    if nx.density(G) == 1:
        for cut_set in combinations(G, len(G) - 1):
            yield set(cut_set)
        return
    seen = []
    H = build_auxiliary_node_connectivity(G)
    H_nodes = H.nodes
    mapping = H.graph['mapping']
    original_H_pred = copy.copy(H._pred)
    R = build_residual_network(H, 'capacity')
    kwargs = {'capacity': 'capacity', 'residual': R}
    if flow_func is None:
        flow_func = default_flow_func
    if flow_func is shortest_augmenting_path:
        kwargs['two_phase'] = True
    if k is None:
        k = nx.node_connectivity(G, flow_func=flow_func)
    X = {n for n, d in sorted(G.degree(), key=itemgetter(1), reverse=True)[:k]}
    if _is_separating_set(G, X):
        seen.append(X)
        yield X
    for x in X:
        non_adjacent = set(G) - X - set(G[x])
        for v in non_adjacent:
            R = flow_func(H, f'{mapping[x]}B', f'{mapping[v]}A', **kwargs)
            flow_value = R.graph['flow_value']
            if flow_value == k:
                E1 = flowed_edges = [(u, w) for u, w, d in R.edges(data=True) if d['flow'] != 0]
                VE1 = incident_nodes = {n for edge in E1 for n in edge}
                saturated_edges = [(u, w, d) for u, w, d in R.edges(data=True) if d['capacity'] == d['flow'] or d['capacity'] == 0]
                R.remove_edges_from(saturated_edges)
                R_closure = nx.transitive_closure(R)
                L = nx.condensation(R)
                cmap = L.graph['mapping']
                inv_cmap = defaultdict(list)
                for n, scc in cmap.items():
                    inv_cmap[scc].append(n)
                VE1 = {cmap[n] for n in VE1}
                for antichain in nx.antichains(L):
                    if not set(antichain).issubset(VE1):
                        continue
                    S = set()
                    for scc in antichain:
                        S.update(inv_cmap[scc])
                    S_ancestors = set()
                    for n in S:
                        S_ancestors.update(R_closure._pred[n])
                    S.update(S_ancestors)
                    if f'{mapping[x]}B' not in S or f'{mapping[v]}A' in S:
                        continue
                    cutset = set()
                    for u in S:
                        cutset.update(((u, w) for w in original_H_pred[u] if w not in S))
                    if any((H_nodes[u]['id'] != H_nodes[w]['id'] for u, w in cutset)):
                        continue
                    node_cut = {H_nodes[u]['id'] for u, _ in cutset}
                    if len(node_cut) == k:
                        if x in node_cut or v in node_cut:
                            continue
                        if node_cut not in seen:
                            yield node_cut
                            seen.append(node_cut)
                H.add_edge(f'{mapping[x]}B', f'{mapping[v]}A', capacity=1)
                H.add_edge(f'{mapping[v]}B', f'{mapping[x]}A', capacity=1)
                R.add_edge(f'{mapping[x]}B', f'{mapping[v]}A', capacity=1)
                R.add_edge(f'{mapping[v]}A', f'{mapping[x]}B', capacity=0)
                R.add_edge(f'{mapping[v]}B', f'{mapping[x]}A', capacity=1)
                R.add_edge(f'{mapping[x]}A', f'{mapping[v]}B', capacity=0)
                R.add_edges_from(saturated_edges)