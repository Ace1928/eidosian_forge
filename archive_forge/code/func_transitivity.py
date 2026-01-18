from collections import Counter
from itertools import chain, combinations
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch
def transitivity(G):
    """Compute graph transitivity, the fraction of all possible triangles
    present in G.

    Possible triangles are identified by the number of "triads"
    (two edges with a shared vertex).

    The transitivity is

    .. math::

        T = 3\\frac{\\#triangles}{\\#triads}.

    Parameters
    ----------
    G : graph

    Returns
    -------
    out : float
       Transitivity

    Examples
    --------
    >>> G = nx.complete_graph(5)
    >>> print(nx.transitivity(G))
    1.0
    """
    triangles_contri = [(t, d * (d - 1)) for v, d, t, _ in _triangles_and_degree_iter(G)]
    if len(triangles_contri) == 0:
        return 0
    triangles, contri = map(sum, zip(*triangles_contri))
    return 0 if triangles == 0 else triangles / contri