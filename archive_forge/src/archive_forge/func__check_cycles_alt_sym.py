from sympy.combinatorics.permutations import Permutation, _af_invert, _af_rmul
from sympy.ntheory import isprime
def _check_cycles_alt_sym(perm):
    """
    Checks for cycles of prime length p with n/2 < p < n-2.

    Explanation
    ===========

    Here `n` is the degree of the permutation. This is a helper function for
    the function is_alt_sym from sympy.combinatorics.perm_groups.

    Examples
    ========

    >>> from sympy.combinatorics.util import _check_cycles_alt_sym
    >>> from sympy.combinatorics import Permutation
    >>> a = Permutation([[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [11, 12]])
    >>> _check_cycles_alt_sym(a)
    False
    >>> b = Permutation([[0, 1, 2, 3, 4, 5, 6], [7, 8, 9, 10]])
    >>> _check_cycles_alt_sym(b)
    True

    See Also
    ========

    sympy.combinatorics.perm_groups.PermutationGroup.is_alt_sym

    """
    n = perm.size
    af = perm.array_form
    current_len = 0
    total_len = 0
    used = set()
    for i in range(n // 2):
        if i not in used and i < n // 2 - total_len:
            current_len = 1
            used.add(i)
            j = i
            while af[j] != i:
                current_len += 1
                j = af[j]
                used.add(j)
            total_len += current_len
            if current_len > n // 2 and current_len < n - 2 and isprime(current_len):
                return True
    return False