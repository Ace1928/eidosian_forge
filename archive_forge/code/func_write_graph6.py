from itertools import islice
import networkx as nx
from networkx.exception import NetworkXError
from networkx.utils import not_implemented_for, open_file
@not_implemented_for('directed')
@not_implemented_for('multigraph')
@open_file(1, mode='wb')
def write_graph6(G, path, nodes=None, header=True):
    """Write a simple undirected graph to a path in graph6 format.

    Parameters
    ----------
    G : Graph (undirected)

    path : str
       The path naming the file to which to write the graph.

    nodes: list or iterable
       Nodes are labeled 0...n-1 in the order provided.  If None the ordering
       given by ``G.nodes()`` is used.

    header: bool
       If True add '>>graph6<<' string to head of data

    Raises
    ------
    NetworkXNotImplemented
        If the graph is directed or is a multigraph.

    ValueError
        If the graph has at least ``2 ** 36`` nodes; the graph6 format
        is only defined for graphs of order less than ``2 ** 36``.

    Examples
    --------
    You can write a graph6 file by giving the path to a file::

        >>> import tempfile
        >>> with tempfile.NamedTemporaryFile(delete=False) as f:
        ...     nx.write_graph6(nx.path_graph(2), f.name)
        ...     _ = f.seek(0)
        ...     print(f.read())
        b'>>graph6<<A_\\n'

    See Also
    --------
    from_graph6_bytes, read_graph6

    Notes
    -----
    The function writes a newline character after writing the encoding
    of the graph.

    The format does not support edge or node labels, parallel edges or
    self loops.  If self loops are present they are silently ignored.

    References
    ----------
    .. [1] Graph6 specification
           <http://users.cecs.anu.edu.au/~bdm/data/formats.html>

    """
    return write_graph6_file(G, path, nodes=nodes, header=header)