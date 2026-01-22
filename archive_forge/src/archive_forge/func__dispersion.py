from itertools import combinations
import networkx as nx
def _dispersion(G_u, u, v):
    """dispersion for all nodes 'v' in a ego network G_u of node 'u'"""
    u_nbrs = set(G_u[u])
    ST = {n for n in G_u[v] if n in u_nbrs}
    set_uv = {u, v}
    possib = combinations(ST, 2)
    total = 0
    for s, t in possib:
        nbrs_s = u_nbrs.intersection(G_u[s]) - set_uv
        if t not in nbrs_s:
            if nbrs_s.isdisjoint(G_u[t]):
                total += 1
    embeddedness = len(ST)
    dispersion_val = total
    if normalized:
        dispersion_val = (total + b) ** alpha
        if embeddedness + c != 0:
            dispersion_val /= embeddedness + c
    return dispersion_val