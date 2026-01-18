from itertools import chain
import networkx as nx
@nx._dispatch(edge_attrs='weight')
def volume(G, S, weight=None):
    """Returns the volume of a set of nodes.

    The *volume* of a set *S* is the sum of the (out-)degrees of nodes
    in *S* (taking into account parallel edges in multigraphs). [1]

    Parameters
    ----------
    G : NetworkX graph

    S : collection
        A collection of nodes in `G`.

    weight : object
        Edge attribute key to use as weight. If not specified, edges
        have weight one.

    Returns
    -------
    number
        The volume of the set of nodes represented by `S` in the graph
        `G`.

    See also
    --------
    conductance
    cut_size
    edge_expansion
    edge_boundary
    normalized_cut_size

    References
    ----------
    .. [1] David Gleich.
           *Hierarchical Directed Spectral Graph Partitioning*.
           <https://www.cs.purdue.edu/homes/dgleich/publications/Gleich%202005%20-%20hierarchical%20directed%20spectral.pdf>

    """
    degree = G.out_degree if G.is_directed() else G.degree
    return sum((d for v, d in degree(S, weight=weight)))