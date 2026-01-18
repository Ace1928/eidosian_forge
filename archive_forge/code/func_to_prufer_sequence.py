from collections import Counter
from itertools import chain
import networkx as nx
from networkx.utils import not_implemented_for
@not_implemented_for('directed')
@nx._dispatch(graphs='T')
def to_prufer_sequence(T):
    """Returns the Prüfer sequence of the given tree.

    A *Prüfer sequence* is a list of *n* - 2 numbers between 0 and
    *n* - 1, inclusive. The tree corresponding to a given Prüfer
    sequence can be recovered by repeatedly joining a node in the
    sequence with a node with the smallest potential degree according to
    the sequence.

    Parameters
    ----------
    T : NetworkX graph
        An undirected graph object representing a tree.

    Returns
    -------
    list
        The Prüfer sequence of the given tree.

    Raises
    ------
    NetworkXPointlessConcept
        If the number of nodes in `T` is less than two.

    NotATree
        If `T` is not a tree.

    KeyError
        If the set of nodes in `T` is not {0, …, *n* - 1}.

    Notes
    -----
    There is a bijection from labeled trees to Prüfer sequences. This
    function is the inverse of the :func:`from_prufer_sequence`
    function.

    Sometimes Prüfer sequences use nodes labeled from 1 to *n* instead
    of from 0 to *n* - 1. This function requires nodes to be labeled in
    the latter form. You can use :func:`~networkx.relabel_nodes` to
    relabel the nodes of your tree to the appropriate format.

    This implementation is from [1]_ and has a running time of
    $O(n)$.

    See also
    --------
    to_nested_tuple
    from_prufer_sequence

    References
    ----------
    .. [1] Wang, Xiaodong, Lei Wang, and Yingjie Wu.
           "An optimal algorithm for Prufer codes."
           *Journal of Software Engineering and Applications* 2.02 (2009): 111.
           <https://doi.org/10.4236/jsea.2009.22016>

    Examples
    --------
    There is a bijection between Prüfer sequences and labeled trees, so
    this function is the inverse of the :func:`from_prufer_sequence`
    function:

    >>> edges = [(0, 3), (1, 3), (2, 3), (3, 4), (4, 5)]
    >>> tree = nx.Graph(edges)
    >>> sequence = nx.to_prufer_sequence(tree)
    >>> sequence
    [3, 3, 3, 4]
    >>> tree2 = nx.from_prufer_sequence(sequence)
    >>> list(tree2.edges()) == edges
    True

    """
    n = len(T)
    if n < 2:
        msg = 'Prüfer sequence undefined for trees with fewer than two nodes'
        raise nx.NetworkXPointlessConcept(msg)
    if not nx.is_tree(T):
        raise nx.NotATree('provided graph is not a tree')
    if set(T) != set(range(n)):
        raise KeyError('tree must have node labels {0, ..., n - 1}')
    degree = dict(T.degree())

    def parents(u):
        return next((v for v in T[u] if degree[v] > 1))
    index = u = next((k for k in range(n) if degree[k] == 1))
    result = []
    for i in range(n - 2):
        v = parents(u)
        result.append(v)
        degree[v] -= 1
        if v < index and degree[v] == 1:
            u = v
        else:
            index = u = next((k for k in range(index + 1, n) if degree[k] == 1))
    return result