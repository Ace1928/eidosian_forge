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
def _galois_group_degree_6_lookup(T, max_tries=30, randomize=False):
    """
    Compute the Galois group of a polynomial of degree 6.

    Explanation
    ===========

    Based on Alg 6.3.10 of [1], but uses resolvent coeff lookup.

    """
    from sympy.combinatorics.galois import S6TransitiveSubgroups
    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 1)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries, history=history, fixed_order=not randomize)
    else:
        raise MaxTriesException
    fl = dup_factor_list(R_dup, ZZ)
    factors_by_deg = defaultdict(list)
    for r, _ in fl[1]:
        factors_by_deg[len(r) - 1].append(r)
    L = sorted(sum([[d] * len(ff) for d, ff in factors_by_deg.items()], []))
    T_has_sq_disc = has_square_disc(T)
    if L == [1, 2, 3]:
        f1 = factors_by_deg[3][0]
        return (S6TransitiveSubgroups.C6, False) if has_square_disc(f1) else (S6TransitiveSubgroups.D6, False)
    elif L == [3, 3]:
        f1, f2 = factors_by_deg[3]
        any_square = has_square_disc(f1) or has_square_disc(f2)
        return (S6TransitiveSubgroups.G18, False) if any_square else (S6TransitiveSubgroups.G36m, False)
    elif L == [2, 4]:
        if T_has_sq_disc:
            return (S6TransitiveSubgroups.S4p, True)
        else:
            f1 = factors_by_deg[4][0]
            return (S6TransitiveSubgroups.A4xC2, False) if has_square_disc(f1) else (S6TransitiveSubgroups.S4xC2, False)
    elif L == [1, 1, 4]:
        return (S6TransitiveSubgroups.A4, True) if T_has_sq_disc else (S6TransitiveSubgroups.S4m, False)
    elif L == [1, 5]:
        return (S6TransitiveSubgroups.PSL2F5, True) if T_has_sq_disc else (S6TransitiveSubgroups.PGL2F5, False)
    elif L == [1, 1, 1, 3]:
        return (S6TransitiveSubgroups.S3, False)
    assert L == [6]
    history = set()
    for i in range(max_tries):
        R_dup = get_resolvent_by_lookup(T, 2)
        if dup_sqf_p(R_dup, ZZ):
            break
        _, T = tschirnhausen_transformation(T, max_tries=max_tries, history=history, fixed_order=not randomize)
    else:
        raise MaxTriesException
    T_has_sq_disc = has_square_disc(T)
    if dup_irreducible_p(R_dup, ZZ):
        return (S6TransitiveSubgroups.A6, True) if T_has_sq_disc else (S6TransitiveSubgroups.S6, False)
    else:
        return (S6TransitiveSubgroups.G36p, True) if T_has_sq_disc else (S6TransitiveSubgroups.G72, False)