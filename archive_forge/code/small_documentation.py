from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (

    Returns the Tutte graph.

    The Tutte graph is a cubic polyhedral, non-Hamiltonian graph. It has
    46 nodes and 69 edges.
    It is a counterexample to Tait's conjecture that every 3-regular polyhedron
    has a Hamiltonian cycle.
    It can be realized geometrically from a tetrahedron by multiply truncating
    three of its vertices [1]_.

    Parameters
    ----------
    create_using : NetworkX graph constructor, optional (default=nx.Graph)
       Graph type to create. If graph instance, then cleared before populated.

    Returns
    -------
    G : networkx Graph
        Tutte graph

    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Tutte_graph
    