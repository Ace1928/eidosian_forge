from collections import defaultdict
import random
from sympy.core.symbol import Dummy, symbols
from sympy.ntheory.primetest import is_square
from sympy.polys.domains import ZZ
from sympy.polys.densebasic import dup_random
from sympy.polys.densetools import dup_eval
from sympy.polys.euclidtools import dup_discriminant
from sympy.polys.factortools import dup_factor_list, dup_irreducible_p
from sympy.polys.numberfields.galois_resolvents import (
from sympy.polys.numberfields.utilities import coeff_search
from sympy.polys.polytools import (Poly, poly_from_expr,
from sympy.polys.sqfreetools import dup_sqf_p
from sympy.utilities import public
def _galois_group_degree_4_lookup(T, max_tries=30, randomize=False):
    """
    Compute the Galois group of a polynomial of degree 4.

    Explanation
    ===========

    Based on Alg 6.3.6 of [1], but uses resolvent coeff lookup.

    """
    from sympy.combinatorics.galois import S4TransitiveSubgroups
    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 0)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries, history=history, fixed_order=not randomize)
    else:
        raise MaxTriesException
    fl = dup_factor_list(R_dup, ZZ)
    L = sorted(sum([[len(r) - 1] * e for r, e in fl[1]], []))
    if L == [6]:
        return (S4TransitiveSubgroups.A4, True) if has_square_disc(T) else (S4TransitiveSubgroups.S4, False)
    if L == [1, 1, 4]:
        return (S4TransitiveSubgroups.C4, False)
    if L == [2, 2, 2]:
        return (S4TransitiveSubgroups.V, True)
    assert L == [2, 4]
    return (S4TransitiveSubgroups.D4, False)