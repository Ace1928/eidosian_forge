import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
def _random_unlabeled_tree(n, cache_trees, cache_forests, seed):
    """Returns a tree on `n` nodes drawn uniformly at random.
    It implements the Wilf's algorithm "Free" of [1]_.

    Parameters
    ----------
    n : int
        The number of nodes, greater than zero.
    cache_trees : list of ints
        Cache for :func:`_num_rooted_trees`.
    cache_forests : list of ints
        Cache for :func:`_num_rooted_forests`.
    seed : random_state
        Indicator of random number generation state.
        See :ref:`Randomness<randomness>`

    Returns
    -------
    (edges, n)
        The tree as a list of edges and number of nodes.

    References
    ----------
    .. [1] Wilf, Herbert S. "The uniform selection of free trees."
        Journal of Algorithms 2.2 (1981): 204-207.
        https://doi.org/10.1016/0196-6774(81)90021-3
    """
    if n % 2 == 1:
        p = 0
    else:
        p = comb(_num_rooted_trees(n // 2, cache_trees) + 1, 2)
    if seed.randint(0, _num_trees(n, cache_trees) - 1) < p:
        return _bicenter(n, cache_trees, seed)
    else:
        f, n_f, r = _random_unlabeled_rooted_forest(n - 1, (n - 1) // 2, cache_trees, cache_forests, seed)
        for i in r:
            f.append((i, n_f))
        return (f, n_f + 1)