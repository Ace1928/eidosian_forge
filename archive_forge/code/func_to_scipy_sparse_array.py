import itertools
from collections import defaultdict
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch(edge_attrs='weight')
def to_scipy_sparse_array(G, nodelist=None, dtype=None, weight='weight', format='csr'):
    """Returns the graph adjacency matrix as a SciPy sparse array.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the sparse matrix.

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in `nodelist`.
       If `nodelist` is None, then the ordering is produced by G.nodes().

    dtype : NumPy data-type, optional
        A valid NumPy dtype used to initialize the array. If None, then the
        NumPy default is used.

    weight : string or None   optional (default='weight')
        The edge attribute that holds the numerical value used for
        the edge weight.  If None then all edge weights are 1.

    format : str in {'bsr', 'csr', 'csc', 'coo', 'lil', 'dia', 'dok'}
        The type of the matrix to be returned (default 'csr').  For
        some algorithms different implementations of sparse matrices
        can perform better.  See [1]_ for details.

    Returns
    -------
    A : SciPy sparse array
       Graph adjacency matrix.

    Notes
    -----
    For directed graphs, matrix entry i,j corresponds to an edge from i to j.

    The matrix entries are populated using the edge attribute held in
    parameter weight. When an edge does not have that attribute, the
    value of the entry is 1.

    For multiple edges the matrix values are the sums of the edge weights.

    When `nodelist` does not contain every node in `G`, the adjacency matrix
    is built from the subgraph of `G` that is induced by the nodes in
    `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal matrix entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute).  If the
    alternate convention of doubling the edge weight is desired the
    resulting SciPy sparse array can be modified as follows:

    >>> G = nx.Graph([(1, 1)])
    >>> A = nx.to_scipy_sparse_array(G)
    >>> print(A.todense())
    [[1]]
    >>> A.setdiag(A.diagonal() * 2)
    >>> print(A.toarray())
    [[2]]

    Examples
    --------
    >>> G = nx.MultiDiGraph()
    >>> G.add_edge(0, 1, weight=2)
    0
    >>> G.add_edge(1, 0)
    0
    >>> G.add_edge(2, 2, weight=3)
    0
    >>> G.add_edge(2, 2)
    1
    >>> S = nx.to_scipy_sparse_array(G, nodelist=[0, 1, 2])
    >>> print(S.toarray())
    [[0 2 0]
     [1 0 0]
     [0 0 4]]

    References
    ----------
    .. [1] Scipy Dev. References, "Sparse Matrices",
       https://docs.scipy.org/doc/scipy/reference/sparse.html
    """
    import scipy as sp
    if len(G) == 0:
        raise nx.NetworkXError('Graph has no nodes or edges')
    if nodelist is None:
        nodelist = list(G)
        nlen = len(G)
    else:
        nlen = len(nodelist)
        if nlen == 0:
            raise nx.NetworkXError('nodelist has no nodes')
        nodeset = set(G.nbunch_iter(nodelist))
        if nlen != len(nodeset):
            for n in nodelist:
                if n not in G:
                    raise nx.NetworkXError(f'Node {n} in nodelist is not in G')
            raise nx.NetworkXError('nodelist contains duplicates.')
        if nlen < len(G):
            G = G.subgraph(nodelist)
    index = dict(zip(nodelist, range(nlen)))
    coefficients = zip(*((index[u], index[v], wt) for u, v, wt in G.edges(data=weight, default=1)))
    try:
        row, col, data = coefficients
    except ValueError:
        row, col, data = ([], [], [])
    if G.is_directed():
        A = sp.sparse.coo_array((data, (row, col)), shape=(nlen, nlen), dtype=dtype)
    else:
        d = data + data
        r = row + col
        c = col + row
        selfloops = list(nx.selfloop_edges(G, data=weight, default=1))
        if selfloops:
            diag_index, diag_data = zip(*((index[u], -wt) for u, v, wt in selfloops))
            d += diag_data
            r += diag_index
            c += diag_index
        A = sp.sparse.coo_array((d, (r, c)), shape=(nlen, nlen), dtype=dtype)
    try:
        return A.asformat(format)
    except ValueError as err:
        raise nx.NetworkXError(f'Unknown sparse matrix format: {format}') from err