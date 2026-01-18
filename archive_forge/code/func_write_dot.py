import os
import tempfile
import networkx as nx
def write_dot(G, path):
    """Write NetworkX graph G to Graphviz dot format on path.

    Parameters
    ----------
    G : graph
       A networkx graph
    path : filename
       Filename or file handle to write

    Notes
    -----
    To use a specific graph layout, call ``A.layout`` prior to `write_dot`.
    Note that some graphviz layouts are not guaranteed to be deterministic,
    see https://gitlab.com/graphviz/graphviz/-/issues/1767 for more info.
    """
    A = to_agraph(G)
    A.write(path)
    A.clear()
    return