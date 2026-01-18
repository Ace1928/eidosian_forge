import networkx as nx
from networkx.exception import NetworkXError
from networkx.readwrite.graph6 import data_to_n, n_to_data
from networkx.utils import not_implemented_for, open_file
def to_sparse6_bytes(G, nodes=None, header=True):
    """Convert an undirected graph to bytes in sparse6 format.

    Parameters
    ----------
    G : Graph (undirected)

    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by ``G.nodes()`` is used.

    header: bool
       If True add '>>sparse6<<' bytes to head of data.

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed.

    ValueError
        If the graph has at least ``2 ** 36`` nodes; the sparse6 format
        is only defined for graphs of order less than ``2 ** 36``.

    Examples
    --------
    >>> nx.to_sparse6_bytes(nx.path_graph(2))
    b'>>sparse6<<:An\\n'

    See Also
    --------
    to_sparse6_bytes, read_sparse6, write_sparse6_bytes

    Notes
    -----
    The returned bytes end with a newline character.

    The format does not support edge or node labels.

    References
    ----------
    .. [1] Graph6 specification
           <https://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    if nodes is not None:
        G = G.subgraph(nodes)
    G = nx.convert_node_labels_to_integers(G, ordering='sorted')
    return b''.join(_generate_sparse6_bytes(G, nodes, header))