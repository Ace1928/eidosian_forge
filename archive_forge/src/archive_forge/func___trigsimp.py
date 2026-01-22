from collections import defaultdict
from functools import reduce
from sympy.core import (sympify, Basic, S, Expr, factor_terms,
from sympy.core.cache import cacheit
from sympy.core.function import (count_ops, _mexpand, FunctionClass, expand,
from sympy.core.numbers import I, Integer, igcd
from sympy.core.sorting import _nodes
from sympy.core.symbol import Dummy, symbols, Wild
from sympy.external.gmpy import SYMPY_INTS
from sympy.functions import sin, cos, exp, cosh, tanh, sinh, tan, cot, coth
from sympy.functions import atan2
from sympy.functions.elementary.hyperbolic import HyperbolicFunction
from sympy.functions.elementary.trigonometric import TrigonometricFunction
from sympy.polys import Poly, factor, cancel, parallel_poly_from_expr
from sympy.polys.domains import ZZ
from sympy.polys.polyerrors import PolificationFailed
from sympy.polys.polytools import groebner
from sympy.simplify.cse_main import cse
from sympy.strategies.core import identity
from sympy.strategies.tree import greedy
from sympy.utilities.iterables import iterable
from sympy.utilities.misc import debug
@cacheit
def __trigsimp(expr, deep=False):
    """recursive helper for trigsimp"""
    from sympy.simplify.fu import TR10i
    if _trigpat is None:
        _trigpats()
    a, b, c, d, matchers_division, matchers_add, matchers_identity, artifacts = _trigpat
    if expr.is_Mul:
        if not expr.is_commutative:
            com, nc = expr.args_cnc()
            expr = _trigsimp(Mul._from_args(com), deep) * Mul._from_args(nc)
        else:
            for i, (pattern, simp, ok1, ok2) in enumerate(matchers_division):
                if not _dotrig(expr, pattern):
                    continue
                newexpr = _match_div_rewrite(expr, i)
                if newexpr is not None:
                    if newexpr != expr:
                        expr = newexpr
                        break
                    else:
                        continue
                res = expr.match(pattern)
                if res and res.get(c, 0):
                    if not res[c].is_integer:
                        ok = ok1.subs(res)
                        if not ok.is_positive:
                            continue
                        ok = ok2.subs(res)
                        if not ok.is_positive:
                            continue
                    if any((w.args[0] == res[b] for w in res[a].atoms(TrigonometricFunction, HyperbolicFunction))):
                        continue
                    expr = simp.subs(res)
                    break
    if expr.is_Add:
        args = []
        for term in expr.args:
            if not term.is_commutative:
                com, nc = term.args_cnc()
                nc = Mul._from_args(nc)
                term = Mul._from_args(com)
            else:
                nc = S.One
            term = _trigsimp(term, deep)
            for pattern, result in matchers_identity:
                res = term.match(pattern)
                if res is not None:
                    term = result.subs(res)
                    break
            args.append(term * nc)
        if args != expr.args:
            expr = Add(*args)
            expr = min(expr, expand(expr), key=count_ops)
        if expr.is_Add:
            for pattern, result in matchers_add:
                if not _dotrig(expr, pattern):
                    continue
                expr = TR10i(expr)
                if expr.has(HyperbolicFunction):
                    res = expr.match(pattern)
                    if res is None or not (a in res and b in res) or any((w.args[0] in (res[a], res[b]) for w in res[d].atoms(TrigonometricFunction, HyperbolicFunction))):
                        continue
                    expr = result.subs(res)
                    break
        for pattern, result, ex in artifacts:
            if not _dotrig(expr, pattern):
                continue
            a_t = Wild('a', exclude=[ex])
            pattern = pattern.subs(a, a_t)
            result = result.subs(a, a_t)
            m = expr.match(pattern)
            was = None
            while m and was != expr:
                was = expr
                if m[a_t] == 0 or -m[a_t] in m[c].args or m[a_t] + m[c] == 0:
                    break
                if d in m and m[a_t] * m[d] + m[c] == 0:
                    break
                expr = result.subs(m)
                m = expr.match(pattern)
                m.setdefault(c, S.Zero)
    elif expr.is_Mul or expr.is_Pow or (deep and expr.args):
        expr = expr.func(*[_trigsimp(a, deep) for a in expr.args])
    try:
        if not expr.has(*_trigs):
            raise TypeError
        e = expr.atoms(exp)
        new = expr.rewrite(exp, deep=deep)
        if new == e:
            raise TypeError
        fnew = factor(new)
        if fnew != new:
            new = sorted([new, factor(new)], key=count_ops)[0]
        if not new.atoms(exp) - e:
            expr = new
    except TypeError:
        pass
    return expr