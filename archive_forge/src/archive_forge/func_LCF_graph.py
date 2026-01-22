from functools import wraps
import networkx as nx
from networkx.exception import NetworkXError
from networkx.generators.classic import (
@nx._dispatch(graphs=None)
def LCF_graph(n, shift_list, repeats, create_using=None):
    """
    Return the cubic graph specified in LCF notation.

    LCF notation (LCF=Lederberg-Coxeter-Fruchte) is a compressed
    notation used in the generation of various cubic Hamiltonian
    graphs of high symmetry. See, for example, dodecahedral_graph,
    desargues_graph, heawood_graph and pappus_graph below.

    n (number of nodes)
      The starting graph is the n-cycle with nodes 0,...,n-1.
      (The null graph is returned if n < 0.)

    shift_list = [s1,s2,..,sk], a list of integer shifts mod n,

    repeats
      integer specifying the number of times that shifts in shift_list
      are successively applied to each v_current in the n-cycle
      to generate an edge between v_current and v_current+shift mod n.

    For v1 cycling through the n-cycle a total of k*repeats
    with shift cycling through shiftlist repeats times connect
    v1 with v1+shift mod n

    The utility graph $K_{3,3}$

    >>> G = nx.LCF_graph(6, [3, -3], 3)

    The Heawood graph

    >>> G = nx.LCF_graph(14, [5, -5], 7)

    See http://mathworld.wolfram.com/LCFNotation.html for a description
    and references.

    """
    if n <= 0:
        return empty_graph(0, create_using)
    G = cycle_graph(n, create_using)
    if G.is_directed():
        raise NetworkXError('Directed Graph not supported')
    G.name = 'LCF_graph'
    nodes = sorted(G)
    n_extra_edges = repeats * len(shift_list)
    if n_extra_edges < 1:
        return G
    for i in range(n_extra_edges):
        shift = shift_list[i % len(shift_list)]
        v1 = nodes[i % n]
        v2 = nodes[(i + shift) % n]
        G.add_edge(v1, v2)
    return G