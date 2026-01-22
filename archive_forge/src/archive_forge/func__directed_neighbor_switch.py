import networkx as nx
from networkx.utils import py_random_state
def _directed_neighbor_switch(G, w, unsat, h_node_residual_out, chords, h_partition_in, partition):
    """Releases one free stub for node w, while preserving joint degree in G.

    Parameters
    ----------
    G : networkx directed graph
        graph within which the edge swap will take place.
    w : integer
        node id for which we need to perform a neighbor switch.
    unsat: set of integers
        set of node ids that have the same degree as w and are unsaturated.
    h_node_residual_out: dict of integers
        for a given node, keeps track of the remaining stubs to be added.
    chords: set of tuples
        keeps track of available positions to add edges.
    h_partition_in: dict of integers
        for a given node, keeps track of its partition id (in degree).
    partition: integer
        partition id to check if chords have to be updated.

    Notes
    -----
    First, it selects node w_prime that (1) has the same degree as w and
    (2) is unsaturated. Then, it selects node v, a neighbor of w, that is
    not connected to w_prime and does an edge swap i.e. removes (w,v) and
    adds (w_prime,v). If neighbor switch is not possible for w using
    w_prime and v, then return w_prime; in [1] it's proven that
    such unsaturated nodes can be used.

    References
    ----------
    [1] B. Tillman, A. Markopoulou, C. T. Butts & M. Gjoka,
        "Construction of Directed 2K Graphs". In Proc. of KDD 2017.
    """
    w_prime = unsat.pop()
    unsat.add(w_prime)
    w_neighbs = list(G.successors(w))
    w_prime_neighbs = list(G.successors(w_prime))
    for v in w_neighbs:
        if v not in w_prime_neighbs and w_prime != v:
            G.remove_edge(w, v)
            G.add_edge(w_prime, v)
            if h_partition_in[v] == partition:
                chords.add((w, v))
                chords.discard((w_prime, v))
            h_node_residual_out[w] += 1
            h_node_residual_out[w_prime] -= 1
            if h_node_residual_out[w_prime] == 0:
                unsat.remove(w_prime)
            return None
    return w_prime