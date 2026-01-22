import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
def _num_rooted_forests(n, q, cache_forests):
    """Returns the number of unlabeled rooted forests with `n` nodes, and with
    no more than `q` nodes per tree. A recursive formula for this is (2) in
    [1]_. This function is implemented using dynamic programming instead of
    recursion.

    Parameters
    ----------
    n : int
        The number of nodes.
    q : int
        The maximum number of nodes for each tree of the forest.
    cache_forests : list of ints
        The $i$-th element is the number of unlabeled rooted forests with
        $i$ nodes, and with no more than `q` nodes per tree; this is used
        as a cache (and is extended to length `n` + 1 if needed).

    Returns
    -------
    int
        The number of unlabeled rooted forests with `n` nodes with no more than
        `q` nodes per tree.

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    for n_i in range(len(cache_forests), n + 1):
        q_i = min(n_i, q)
        cache_forests.append(sum([d * cache_forests[n_i - j * d] * cache_forests[d - 1] for d in range(1, q_i + 1) for j in range(1, n_i // d + 1)]) // n_i)
    return cache_forests[n]