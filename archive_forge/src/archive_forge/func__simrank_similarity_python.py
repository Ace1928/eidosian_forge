import math
import time
import warnings
from dataclasses import dataclass
from itertools import product
import networkx as nx
def _simrank_similarity_python(G, source=None, target=None, importance_factor=0.9, max_iterations=1000, tolerance=0.0001):
    """Returns the SimRank similarity of nodes in the graph ``G``.

    This pure Python version is provided for pedagogical purposes.

    Examples
    --------
    >>> G = nx.cycle_graph(2)
    >>> nx.similarity._simrank_similarity_python(G)
    {0: {0: 1, 1: 0.0}, 1: {0: 0.0, 1: 1}}
    >>> nx.similarity._simrank_similarity_python(G, source=0)
    {0: 1, 1: 0.0}
    >>> nx.similarity._simrank_similarity_python(G, source=0, target=0)
    1
    """
    newsim = {u: {v: 1 if u == v else 0 for v in G} for u in G}

    def avg_sim(s):
        return sum((newsim[w][x] for w, x in s)) / len(s) if s else 0.0
    Gadj = G.pred if G.is_directed() else G.adj

    def sim(u, v):
        return importance_factor * avg_sim(list(product(Gadj[u], Gadj[v])))
    for its in range(max_iterations):
        oldsim = newsim
        newsim = {u: {v: sim(u, v) if u != v else 1 for v in G} for u in G}
        is_close = all((all((abs(newsim[u][v] - old) <= tolerance * (1 + abs(old)) for v, old in nbrs.items())) for u, nbrs in oldsim.items()))
        if is_close:
            break
    if its + 1 == max_iterations:
        raise nx.ExceededMaxIterations(f'simrank did not converge after {max_iterations} iterations.')
    if source is not None and target is not None:
        return newsim[source][target]
    if source is not None:
        return newsim[source]
    return newsim