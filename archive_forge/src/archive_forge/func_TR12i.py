from collections import defaultdict
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.exprtools import Factors, gcd_terms, factor_terms
from sympy.core.function import expand_mul
from sympy.core.mul import Mul
from sympy.core.numbers import pi, I
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy
from sympy.core.sympify import sympify
from sympy.core.traversal import bottom_up
from sympy.functions.combinatorial.factorials import binomial
from sympy.functions.elementary.hyperbolic import (
from sympy.functions.elementary.trigonometric import (
from sympy.ntheory.factor_ import perfect_power
from sympy.polys.polytools import factor
from sympy.strategies.tree import greedy
from sympy.strategies.core import identity, debug
from sympy import SYMPY_DEBUG
def TR12i(rv):
    """Combine tan arguments as
    (tan(y) + tan(x))/(tan(x)*tan(y) - 1) -> -tan(x + y).

    Examples
    ========

    >>> from sympy.simplify.fu import TR12i
    >>> from sympy import tan
    >>> from sympy.abc import a, b, c
    >>> ta, tb, tc = [tan(i) for i in (a, b, c)]
    >>> TR12i((ta + tb)/(-ta*tb + 1))
    tan(a + b)
    >>> TR12i((ta + tb)/(ta*tb - 1))
    -tan(a + b)
    >>> TR12i((-ta - tb)/(ta*tb - 1))
    tan(a + b)
    >>> eq = (ta + tb)/(-ta*tb + 1)**2*(-3*ta - 3*tc)/(2*(ta*tc - 1))
    >>> TR12i(eq.expand())
    -3*tan(a + b)*tan(a + c)/(2*(tan(a) + tan(b) - 1))
    """

    def f(rv):
        if not (rv.is_Add or rv.is_Mul or rv.is_Pow):
            return rv
        n, d = rv.as_numer_denom()
        if not d.args or not n.args:
            return rv
        dok = {}

        def ok(di):
            m = as_f_sign_1(di)
            if m:
                g, f, s = m
                if s is S.NegativeOne and f.is_Mul and (len(f.args) == 2) and all((isinstance(fi, tan) for fi in f.args)):
                    return (g, f)
        d_args = list(Mul.make_args(d))
        for i, di in enumerate(d_args):
            m = ok(di)
            if m:
                g, t = m
                s = Add(*[_.args[0] for _ in t.args])
                dok[s] = S.One
                d_args[i] = g
                continue
            if di.is_Add:
                di = factor(di)
                if di.is_Mul:
                    d_args.extend(di.args)
                    d_args[i] = S.One
            elif di.is_Pow and (di.exp.is_integer or di.base.is_positive):
                m = ok(di.base)
                if m:
                    g, t = m
                    s = Add(*[_.args[0] for _ in t.args])
                    dok[s] = di.exp
                    d_args[i] = g ** di.exp
                else:
                    di = factor(di)
                    if di.is_Mul:
                        d_args.extend(di.args)
                        d_args[i] = S.One
        if not dok:
            return rv

        def ok(ni):
            if ni.is_Add and len(ni.args) == 2:
                a, b = ni.args
                if isinstance(a, tan) and isinstance(b, tan):
                    return (a, b)
        n_args = list(Mul.make_args(factor_terms(n)))
        hit = False
        for i, ni in enumerate(n_args):
            m = ok(ni)
            if not m:
                m = ok(-ni)
                if m:
                    n_args[i] = S.NegativeOne
                elif ni.is_Add:
                    ni = factor(ni)
                    if ni.is_Mul:
                        n_args.extend(ni.args)
                        n_args[i] = S.One
                    continue
                elif ni.is_Pow and (ni.exp.is_integer or ni.base.is_positive):
                    m = ok(ni.base)
                    if m:
                        n_args[i] = S.One
                    else:
                        ni = factor(ni)
                        if ni.is_Mul:
                            n_args.extend(ni.args)
                            n_args[i] = S.One
                        continue
                else:
                    continue
            else:
                n_args[i] = S.One
            hit = True
            s = Add(*[_.args[0] for _ in m])
            ed = dok[s]
            newed = ed.extract_additively(S.One)
            if newed is not None:
                if newed:
                    dok[s] = newed
                else:
                    dok.pop(s)
            n_args[i] *= -tan(s)
        if hit:
            rv = Mul(*n_args) / Mul(*d_args) / Mul(*[(Add(*[tan(a) for a in i.args]) - 1) ** e for i, e in dok.items()])
        return rv
    return bottom_up(rv, f)