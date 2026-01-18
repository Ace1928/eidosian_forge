from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@_raise_on_directed
@nx._dispatch(graphs=None)
def tutte_graph(create_using=None):
    """
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
    """
    G = nx.from_dict_of_lists({0: [1, 2, 3], 1: [4, 26], 2: [10, 11], 3: [18, 19], 4: [5, 33], 5: [6, 29], 6: [7, 27], 7: [8, 14], 8: [9, 38], 9: [10, 37], 10: [39], 11: [12, 39], 12: [13, 35], 13: [14, 15], 14: [34], 15: [16, 22], 16: [17, 44], 17: [18, 43], 18: [45], 19: [20, 45], 20: [21, 41], 21: [22, 23], 22: [40], 23: [24, 27], 24: [25, 32], 25: [26, 31], 26: [33], 27: [28], 28: [29, 32], 29: [30], 30: [31, 33], 31: [32], 34: [35, 38], 35: [36], 36: [37, 39], 37: [38], 40: [41, 44], 41: [42], 42: [43, 45], 43: [44]}, create_using=create_using)
    G.name = "Tutte's Graph"
    return G