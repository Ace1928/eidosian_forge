from math import sqrt
import networkx as nx
from networkx.utils import py_random_state

    Perform a "swap" operation on a threshold sequence.

    The swap preserves the number of nodes and edges
    in the graph for the given sequence.
    The resulting sequence is still a threshold sequence.

    Perform one split and one combine operation on the
    'd's of a creation sequence for a threshold graph.
    This operation maintains the number of nodes and edges
    in the graph, but shifts the edges from node to node
    maintaining the threshold quality of the graph.

    seed : integer, random_state, or None (default)
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`.
    