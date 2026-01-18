import networkx as nx
from networkx.utils import not_implemented_for
Returns the transition matrix of G.

    This is a row stochastic giving the transition probabilities while
    performing a random walk on the graph. Depending on the value of walk_type,
    P can be the transition matrix induced by a random walk, a lazy random walk,
    or a random walk with teleportation (PageRank).

    Parameters
    ----------
    G : DiGraph
       A NetworkX graph

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in nodelist.
       If nodelist is None, then the ordering is produced by G.nodes().

    weight : string or None, optional (default='weight')
       The edge data key used to compute each value in the matrix.
       If None, then each edge has weight 1.

    walk_type : string or None, optional (default=None)
       If None, `P` is selected depending on the properties of the
       graph. Otherwise is one of 'random', 'lazy', or 'pagerank'

    alpha : real
       (1 - alpha) is the teleportation probability used with pagerank

    Returns
    -------
    P : numpy.ndarray
      transition matrix of G.

    Raises
    ------
    NetworkXError
        If walk_type not specified or alpha not in valid range
    