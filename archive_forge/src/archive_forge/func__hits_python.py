import networkx as nx
def _hits_python(G, max_iter=100, tol=1e-08, nstart=None, normalized=True):
    if isinstance(G, (nx.MultiGraph, nx.MultiDiGraph)):
        raise Exception('hits() not defined for graphs with multiedges.')
    if len(G) == 0:
        return ({}, {})
    if nstart is None:
        h = dict.fromkeys(G, 1.0 / G.number_of_nodes())
    else:
        h = nstart
        s = 1.0 / sum(h.values())
        for k in h:
            h[k] *= s
    for _ in range(max_iter):
        hlast = h
        h = dict.fromkeys(hlast.keys(), 0)
        a = dict.fromkeys(hlast.keys(), 0)
        for n in h:
            for nbr in G[n]:
                a[nbr] += hlast[n] * G[n][nbr].get('weight', 1)
        for n in h:
            for nbr in G[n]:
                h[n] += a[nbr] * G[n][nbr].get('weight', 1)
        s = 1.0 / max(h.values())
        for n in h:
            h[n] *= s
        s = 1.0 / max(a.values())
        for n in a:
            a[n] *= s
        err = sum((abs(h[n] - hlast[n]) for n in h))
        if err < tol:
            break
    else:
        raise nx.PowerIterationFailedConvergence(max_iter)
    if normalized:
        s = 1.0 / sum(a.values())
        for n in a:
            a[n] *= s
        s = 1.0 / sum(h.values())
        for n in h:
            h[n] *= s
    return (h, a)