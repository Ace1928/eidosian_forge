import warnings
from collections import Counter, defaultdict
from math import comb, factorial
import networkx as nx
from networkx.utils import py_random_state
def _select_jd_trees(n, cache_trees, seed):
    """Returns a pair $(j,d)$ with a specific probability

    Given $n$, returns a pair of positive integers $(j,d)$ with the probability
    specified in formula (5) of Chapter 29 of [1]_.

    Parameters
    ----------
    n : int
        The number of nodes
    cache_trees : list of ints
        Cache for :func:`_num_rooted_trees`.
    seed : random_state
       See :ref:`Randomness<randomness>`.

    Returns
    -------
    (int, int)
        A pair of positive integers $(j,d)$ satisfying formula (5) of
        Chapter 29 of [1]_.

    References
    ----------
    .. [1] Nijenhuis, Albert, and Wilf, Herbert S.
        "Combinatorial algorithms: for computers and calculators."
        Academic Press, 1978.
        https://doi.org/10.1016/C2013-0-11243-3
    """
    p = seed.randint(0, _num_rooted_trees(n, cache_trees) * (n - 1) - 1)
    cumsum = 0
    for d in range(n - 1, 0, -1):
        for j in range(1, (n - 1) // d + 1):
            cumsum += d * _num_rooted_trees(n - j * d, cache_trees) * _num_rooted_trees(d, cache_trees)
            if p < cumsum:
                return (j, d)