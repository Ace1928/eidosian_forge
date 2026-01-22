from collections import Counter
from itertools import chain, combinations
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('multigraph')
def _directed_weighted_triangles_and_degree_iter(G, nodes=None, weight='weight'):
    """Return an iterator of
    (node, total_degree, reciprocal_degree, directed_weighted_triangles).

    Used for directed weighted clustering.
    Note that unlike `_weighted_triangles_and_degree_iter()`, this function counts
    directed triangles so does not count triangles twice.

    """
    import numpy as np
    if weight is None or G.number_of_edges() == 0:
        max_weight = 1
    else:
        max_weight = max((d.get(weight, 1) for u, v, d in G.edges(data=True)))
    nodes_nbrs = ((n, G._pred[n], G._succ[n]) for n in G.nbunch_iter(nodes))

    def wt(u, v):
        return G[u][v].get(weight, 1) / max_weight
    for i, preds, succs in nodes_nbrs:
        ipreds = set(preds) - {i}
        isuccs = set(succs) - {i}
        directed_triangles = 0
        for j in ipreds:
            jpreds = set(G._pred[j]) - {j}
            jsuccs = set(G._succ[j]) - {j}
            directed_triangles += sum(np.cbrt([wt(j, i) * wt(k, i) * wt(k, j) for k in ipreds & jpreds]))
            directed_triangles += sum(np.cbrt([wt(j, i) * wt(k, i) * wt(j, k) for k in ipreds & jsuccs]))
            directed_triangles += sum(np.cbrt([wt(j, i) * wt(i, k) * wt(k, j) for k in isuccs & jpreds]))
            directed_triangles += sum(np.cbrt([wt(j, i) * wt(i, k) * wt(j, k) for k in isuccs & jsuccs]))
        for j in isuccs:
            jpreds = set(G._pred[j]) - {j}
            jsuccs = set(G._succ[j]) - {j}
            directed_triangles += sum(np.cbrt([wt(i, j) * wt(k, i) * wt(k, j) for k in ipreds & jpreds]))
            directed_triangles += sum(np.cbrt([wt(i, j) * wt(k, i) * wt(j, k) for k in ipreds & jsuccs]))
            directed_triangles += sum(np.cbrt([wt(i, j) * wt(i, k) * wt(k, j) for k in isuccs & jpreds]))
            directed_triangles += sum(np.cbrt([wt(i, j) * wt(i, k) * wt(j, k) for k in isuccs & jsuccs]))
        dtotal = len(ipreds) + len(isuccs)
        dbidirectional = len(ipreds & isuccs)
        yield (i, dtotal, dbidirectional, directed_triangles)