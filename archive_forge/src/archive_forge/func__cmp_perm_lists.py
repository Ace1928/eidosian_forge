from sympy.combinatorics import Permutation
from sympy.combinatorics.util import _distribute_gens_by_base
def _cmp_perm_lists(first, second):
    """
    Compare two lists of permutations as sets.

    Explanation
    ===========

    This is used for testing purposes. Since the array form of a
    permutation is currently a list, Permutation is not hashable
    and cannot be put into a set.

    Examples
    ========

    >>> from sympy.combinatorics.permutations import Permutation
    >>> from sympy.combinatorics.testutil import _cmp_perm_lists
    >>> a = Permutation([0, 2, 3, 4, 1])
    >>> b = Permutation([1, 2, 0, 4, 3])
    >>> c = Permutation([3, 4, 0, 1, 2])
    >>> ls1 = [a, b, c]
    >>> ls2 = [b, c, a]
    >>> _cmp_perm_lists(ls1, ls2)
    True

    """
    return {tuple(a) for a in first} == {tuple(a) for a in second}