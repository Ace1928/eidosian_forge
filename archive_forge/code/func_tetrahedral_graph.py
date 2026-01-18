from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@nx._dispatch(graphs=None)
def tetrahedral_graph(create_using=None):
    """
    Returns the 3-regular Platonic Tetrahedral graph.

    Tetrahedral graph has 4 nodes and 6 edges. It is a
    special case of the complete graph, K4, and wheel graph, W4.
    It is one of the 5 platonic graphs [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Tetrahedral Graph

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Tetrahedron#Tetrahedral_graph

    """
    G = complete_graph(4, create_using)
    G.name = 'Platonic Tetrahedral Graph'
    return G