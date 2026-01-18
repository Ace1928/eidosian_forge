import networkx as nx
from networkx.exception import NetworkXError
from networkx.readwrite.graph6 import data_to_n, n_to_data
from networkx.utils import not_implemented_for, open_file
@not_implemented_for('directed')
@open_file(1, mode='wb')
def write_sparse6(G, path, nodes=None, header=True):
    """Write graph G to given path in sparse6 format.

    Parameters
    ----------
    G : Graph (undirected)

    path : file or string
       File or filename to write

    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by G.nodes() is used.

    header: bool
       If True add '>>sparse6<<' string to head of data

    Raises
    ------
    NetworkXError
        If the graph is directed

    Examples
    --------
    You can write a sparse6 file by giving the path to the file::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False) as f:
        ...     nx.write_sparse6(nx.path_graph(2), f.name)
        ...     print(f.read())
        b'>>sparse6<<:An\\n'

    You can also write a sparse6 file by giving an open file-like object::

        >>> with tempfile.NamedTemporaryFile() as f:
        ...     nx.write_sparse6(nx.path_graph(2), f)
        ...     _ = f.seek(0)
        ...     print(f.read())
        b'>>sparse6<<:An\\n'

    See Also
    --------
    read_sparse6, from_sparse6_bytes

    Notes
    -----
    The format does not support edge or node labels.

    References
    ----------
    .. [1] Sparse6 specification
           <https://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    if nodes is not None:
        G = G.subgraph(nodes)
    G = nx.convert_node_labels_to_integers(G, ordering='sorted')
    for b in _generate_sparse6_bytes(G, nodes, header):
        path.write(b)