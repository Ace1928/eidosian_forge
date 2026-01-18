import networkx as nx
from networkx.utils import py_random_state
@py_random_state(3)
@nx._dispatch(graphs=None)
def uniform_random_intersection_graph(n, m, p, seed=None):
    """Returns a uniform random intersection graph.

    Parameters
    ----------
    n : int
        The number of nodes in the first bipartite set (nodes)
    m : int
        The number of nodes in the second bipartite set (attributes)
    p : float
        Probability of connecting nodes between bipartite sets
    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.

    See Also
    --------
    gnp_random_graph

    References
    ----------
    .. [1] K.B. Singer-Cohen, Random Intersection Graphs, 1995,
       PhD thesis, Johns Hopkins University
    .. [2] Fill, J. A., Scheinerman, E. R., and Singer-Cohen, K. B.,
       Random intersection graphs when m = !(n):
       An equivalence theorem relating the evolution of the g(n, m, p)
       and g(n, p) models. Random Struct. Algorithms 16, 2 (2000), 156–176.
    """
    from networkx.algorithms import bipartite
    G = bipartite.random_graph(n, m, p, seed)
    return nx.projected_graph(G, range(n))