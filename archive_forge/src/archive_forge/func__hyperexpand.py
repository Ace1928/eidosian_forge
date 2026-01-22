from collections import defaultdict
from itertools import product
from functools import reduce
from math import prod
from sympy import SYMPY_DEBUG
from sympy.core import (S, Dummy, symbols, sympify, Tuple, expand, I, pi, Mul,
from sympy.core.mod import Mod
from sympy.core.sorting import default_sort_key
from sympy.functions import (exp, sqrt, root, log, lowergamma, cos,
from sympy.functions.elementary.complexes import polarify, unpolarify
from sympy.functions.special.hyper import (hyper, HyperRep_atanh,
from sympy.matrices import Matrix, eye, zeros
from sympy.polys import apart, poly, Poly
from sympy.series import residue
from sympy.simplify.powsimp import powdenest
from sympy.utilities.iterables import sift
def _hyperexpand(func, z, ops0=[], z0=Dummy('z0'), premult=1, prem=0, rewrite='default'):
    """
    Try to find an expression for the hypergeometric function ``func``.

    Explanation
    ===========

    The result is expressed in terms of a dummy variable ``z0``. Then it
    is multiplied by ``premult``. Then ``ops0`` is applied.
    ``premult`` must be a*z**prem for some a independent of ``z``.
    """
    if z.is_zero:
        return S.One
    from sympy.simplify.simplify import simplify
    z = polarify(z, subs=False)
    if rewrite == 'default':
        rewrite = 'nonrepsmall'

    def carryout_plan(f, ops):
        C = apply_operators(f.C.subs(f.z, z0), ops, make_derivative_operator(f.M.subs(f.z, z0), z0))
        C = apply_operators(C, ops0, make_derivative_operator(f.M.subs(f.z, z0) + prem * eye(f.M.shape[0]), z0))
        if premult == 1:
            C = C.applyfunc(make_simp(z0))
        r = reduce(lambda s, m: s + m[0] * m[1], zip(C, f.B.subs(f.z, z0)), S.Zero) * premult
        res = r.subs(z0, z)
        if rewrite:
            res = res.rewrite(rewrite)
        return res
    global _collection
    if _collection is None:
        _collection = FormulaCollection()
    debug('Trying to expand hypergeometric function ', func)
    func, ops = reduce_order(func)
    if ops:
        debug('  Reduced order to ', func)
    else:
        debug('  Could not reduce order.')
    res = try_polynomial(func, z0)
    if res is not None:
        debug('  Recognised polynomial.')
        p = apply_operators(res, ops, lambda f: z0 * f.diff(z0))
        p = apply_operators(p * premult, ops0, lambda f: z0 * f.diff(z0))
        return unpolarify(simplify(p).subs(z0, z))
    p = S.Zero
    res = try_shifted_sum(func, z0)
    if res is not None:
        func, nops, p = res
        debug('  Recognised shifted sum, reduced order to ', func)
        ops += nops
    p = apply_operators(p, ops, lambda f: z0 * f.diff(z0))
    p = apply_operators(p * premult, ops0, lambda f: z0 * f.diff(z0))
    p = simplify(p).subs(z0, z)
    if unpolarify(z) in [1, -1] and (len(func.ap), len(func.bq)) == (2, 1):
        f = build_hypergeometric_formula(func)
        r = carryout_plan(f, ops).replace(hyper, hyperexpand_special)
        if not r.has(hyper):
            return r + p
    formula = _collection.lookup_origin(func)
    if formula is None:
        formula = try_lerchphi(func)
    if formula is None:
        debug('  Could not find an origin. ', 'Will return answer in terms of simpler hypergeometric functions.')
        formula = build_hypergeometric_formula(func)
    debug('  Found an origin: ', formula.closed_form, ' ', formula.func)
    ops += devise_plan(func, formula.func, z0)
    r = carryout_plan(formula, ops) + p
    return powdenest(r, polar=True).replace(hyper, hyperexpand_special)