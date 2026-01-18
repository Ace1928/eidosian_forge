import itertools
from collections import defaultdict
import networkx as nx
from networkx.utils import not_implemented_for
@nx._dispatch(edge_attrs='weight')
def to_pandas_adjacency(G, nodelist=None, dtype=None, order=None, multigraph_weight=sum, weight='weight', nonedge=0.0):
    """Returns the graph adjacency matrix as a Pandas DataFrame.

    Parameters
    ----------
    G : graph
        The NetworkX graph used to construct the Pandas DataFrame.

    nodelist : list, optional
       The rows and columns are ordered according to the nodes in `nodelist`.
       If `nodelist` is None, then the ordering is produced by G.nodes().

    multigraph_weight : {sum, min, max}, optional
        An operator that determines how weights in multigraphs are handled.
        The default is to sum the weights of the multiple edges.

    weight : string or None, optional
        The edge attribute that holds the numerical value used for
        the edge weight.  If an edge does not have that attribute, then the
        value 1 is used instead.

    nonedge : float, optional
        The matrix values corresponding to nonedges are typically set to zero.
        However, this could be undesirable if there are matrix values
        corresponding to actual edges that also have the value zero. If so,
        one might prefer nonedges to have some other value, such as nan.

    Returns
    -------
    df : Pandas DataFrame
       Graph adjacency matrix

    Notes
    -----
    For directed graphs, entry i,j corresponds to an edge from i to j.

    The DataFrame entries are assigned to the weight edge attribute. When
    an edge does not have a weight attribute, the value of the entry is set to
    the number 1.  For multiple (parallel) edges, the values of the entries
    are determined by the 'multigraph_weight' parameter.  The default is to
    sum the weight attributes for each of the parallel edges.

    When `nodelist` does not contain every node in `G`, the matrix is built
    from the subgraph of `G` that is induced by the nodes in `nodelist`.

    The convention used for self-loop edges in graphs is to assign the
    diagonal matrix entry value to the weight attribute of the edge
    (or the number 1 if the edge has no weight attribute).  If the
    alternate convention of doubling the edge weight is desired the
    resulting Pandas DataFrame can be modified as follows:

    >>> import pandas as pd
    >>> pd.options.display.max_columns = 20
    >>> import numpy as np
    >>> G = nx.Graph([(1, 1)])
    >>> df = nx.to_pandas_adjacency(G, dtype=int)
    >>> df
       1
    1  1
    >>> df.values[np.diag_indices_from(df)] *= 2
    >>> df
       1
    1  2

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
    >>> nx.to_pandas_adjacency(G, nodelist=[0, 1, 2], dtype=int)
       0  1  2
    0  0  2  0
    1  1  0  0
    2  0  0  4

    """
    import pandas as pd
    M = to_numpy_array(G, nodelist=nodelist, dtype=dtype, order=order, multigraph_weight=multigraph_weight, weight=weight, nonedge=nonedge)
    if nodelist is None:
        nodelist = list(G)
    return pd.DataFrame(data=M, index=nodelist, columns=nodelist)