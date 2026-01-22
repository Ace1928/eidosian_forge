def _dual_spanning_tree(K, T):
    """Returns the spanning tree for the white graph dual to the spanning tree T for the black graph. Here, dual means the edges do not intersect, i.e., no common crossings."""
    G = K.white_graph()
    crossings_to_ignore = [e[2] for e in T.edges()]
    crossings_to_use = [c for c in K.crossings if c not in crossings_to_ignore]
    return Graph([K._crossing_to_edge(G, c) for c in crossings_to_use])