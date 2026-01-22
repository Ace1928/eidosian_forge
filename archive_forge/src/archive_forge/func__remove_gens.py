from sympy.combinatorics.permutations import Permutation, _af_invert, _af_rmul
from sympy.ntheory import isprime
def _remove_gens(base, strong_gens, basic_orbits=None, strong_gens_distr=None):
    """
    Remove redundant generators from a strong generating set.

    Parameters
    ==========

    ``base`` - a base
    ``strong_gens`` - a strong generating set relative to ``base``
    ``basic_orbits`` - basic orbits
    ``strong_gens_distr`` - strong generators distributed by membership in basic
    stabilizers

    Returns
    =======

    A strong generating set with respect to ``base`` which is a subset of
    ``strong_gens``.

    Examples
    ========

    >>> from sympy.combinatorics import SymmetricGroup
    >>> from sympy.combinatorics.util import _remove_gens
    >>> from sympy.combinatorics.testutil import _verify_bsgs
    >>> S = SymmetricGroup(15)
    >>> base, strong_gens = S.schreier_sims_incremental()
    >>> new_gens = _remove_gens(base, strong_gens)
    >>> len(new_gens)
    14
    >>> _verify_bsgs(S, base, new_gens)
    True

    Notes
    =====

    This procedure is outlined in [1],p.95.

    References
    ==========

    .. [1] Holt, D., Eick, B., O'Brien, E.
           "Handbook of computational group theory"

    """
    from sympy.combinatorics.perm_groups import _orbit
    base_len = len(base)
    degree = strong_gens[0].size
    if strong_gens_distr is None:
        strong_gens_distr = _distribute_gens_by_base(base, strong_gens)
    if basic_orbits is None:
        basic_orbits = []
        for i in range(base_len):
            basic_orbit = _orbit(degree, strong_gens_distr[i], base[i])
            basic_orbits.append(basic_orbit)
    strong_gens_distr.append([])
    res = strong_gens[:]
    for i in range(base_len - 1, -1, -1):
        gens_copy = strong_gens_distr[i][:]
        for gen in strong_gens_distr[i]:
            if gen not in strong_gens_distr[i + 1]:
                temp_gens = gens_copy[:]
                temp_gens.remove(gen)
                if temp_gens == []:
                    continue
                temp_orbit = _orbit(degree, temp_gens, base[i])
                if temp_orbit == basic_orbits[i]:
                    gens_copy.remove(gen)
                    res.remove(gen)
    return res