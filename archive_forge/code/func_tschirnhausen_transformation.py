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
def tschirnhausen_transformation(T, max_coeff=10, max_tries=30, history=None, fixed_order=True):
    """
    Given a univariate, monic, irreducible polynomial over the integers, find
    another such polynomial defining the same number field.

    Explanation
    ===========

    See Alg 6.3.4 of [1].

    Parameters
    ==========

    T : Poly
        The given polynomial
    max_coeff : int
        When choosing a transformation as part of the process,
        keep the coeffs between plus and minus this.
    max_tries : int
        Consider at most this many transformations.
    history : set, None, optional (default=None)
        Pass a set of ``Poly.rep``'s in order to prevent any of these
        polynomials from being returned as the polynomial ``U`` i.e. the
        transformation of the given polynomial *T*. The given poly *T* will
        automatically be added to this set, before we try to find a new one.
    fixed_order : bool, default True
        If ``True``, work through candidate transformations A(x) in a fixed
        order, from small coeffs to large, resulting in deterministic behavior.
        If ``False``, the A(x) are chosen randomly, while still working our way
        up from small coefficients to larger ones.

    Returns
    =======

    Pair ``(A, U)``

        ``A`` and ``U`` are ``Poly``, ``A`` is the
        transformation, and ``U`` is the transformed polynomial that defines
        the same number field as *T*. The polynomial ``A`` maps the roots of
        *T* to the roots of ``U``.

    Raises
    ======

    MaxTriesException
        if could not find a polynomial before exceeding *max_tries*.

    """
    X = Dummy('X')
    n = T.degree()
    if history is None:
        history = set()
    history.add(T.rep)
    if fixed_order:
        coeff_generators = {}
        deg_coeff_sum = 3
        current_degree = 2

    def get_coeff_generator(degree):
        gen = coeff_generators.get(degree, coeff_search(degree, 1))
        coeff_generators[degree] = gen
        return gen
    for i in range(max_tries):
        if fixed_order:
            gen = get_coeff_generator(current_degree)
            coeffs = next(gen)
            m = max((abs(c) for c in coeffs))
            if current_degree + m > deg_coeff_sum:
                if current_degree == 2:
                    deg_coeff_sum += 1
                    current_degree = deg_coeff_sum - 1
                else:
                    current_degree -= 1
                gen = get_coeff_generator(current_degree)
                coeffs = next(gen)
            a = [ZZ(1)] + [ZZ(c) for c in coeffs]
        else:
            C = min(i // 5 + 1, max_coeff)
            d = random.randint(2, n - 1)
            a = dup_random(d, -C, C, ZZ)
        A = Poly(a, T.gen)
        U = Poly(T.resultant(X - A), X)
        if U.rep not in history and dup_sqf_p(U.rep.rep, ZZ):
            return (A, U)
    raise MaxTriesException