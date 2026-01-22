from sympy.combinatorics.permutations import Permutation, _af_rmul, \
from sympy.combinatorics.perm_groups import PermutationGroup, _orbit, \
from sympy.combinatorics.util import _distribute_gens_by_base, \
def _min_dummies(dummies, sym, indices):
    """
    Return list of minima of the orbits of indices in group of dummies.
    See ``double_coset_can_rep`` for the description of ``dummies`` and ``sym``.
    ``indices`` is the initial list of dummy indices.

    Examples
    ========

    >>> from sympy.combinatorics.tensor_can import _min_dummies
    >>> _min_dummies([list(range(2, 8))], [0], list(range(10)))
    [0, 1, 2, 2, 2, 2, 2, 2, 8, 9]
    """
    num_types = len(sym)
    m = [min(dx) if dx else None for dx in dummies]
    res = indices[:]
    for i in range(num_types):
        for c, i in enumerate(indices):
            for j in range(num_types):
                if i in dummies[j]:
                    res[c] = m[j]
                    break
    return res