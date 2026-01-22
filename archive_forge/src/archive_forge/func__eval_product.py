from typing import Tuple as tTuple
from .expr_with_intlimits import ExprWithIntLimits
from .summations import Sum, summation, _dummy_with_inherited_properties_concrete
from sympy.core.expr import Expr
from sympy.core.exprtools import factor_terms
from sympy.core.function import Derivative
from sympy.core.mul import Mul
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.functions.combinatorial.factorials import RisingFactorial
from sympy.functions.elementary.exponential import exp, log
from sympy.functions.special.tensor_functions import KroneckerDelta
from sympy.polys import quo, roots
def _eval_product(self, term, limits):
    k, a, n = limits
    if k not in term.free_symbols:
        if (term - 1).is_zero:
            return S.One
        return term ** (n - a + 1)
    if a == n:
        return term.subs(k, a)
    from .delta import deltaproduct, _has_simple_delta
    if term.has(KroneckerDelta) and _has_simple_delta(term, limits[0]):
        return deltaproduct(term, limits)
    dif = n - a
    definite = dif.is_Integer
    if definite and dif < 100:
        return self._eval_product_direct(term, limits)
    elif term.is_polynomial(k):
        poly = term.as_poly(k)
        A = B = Q = S.One
        all_roots = roots(poly)
        M = 0
        for r, m in all_roots.items():
            M += m
            A *= RisingFactorial(a - r, n - a + 1) ** m
            Q *= (n - r) ** m
        if M < poly.degree():
            arg = quo(poly, Q.as_poly(k))
            B = self.func(arg, (k, a, n)).doit()
        return poly.LC() ** (n - a + 1) * A * B
    elif term.is_Add:
        factored = factor_terms(term, fraction=True)
        if factored.is_Mul:
            return self._eval_product(factored, (k, a, n))
    elif term.is_Mul:
        without_k, with_k = term.as_coeff_mul(k)
        if len(with_k) >= 2:
            exclude, include = ([], [])
            for t in with_k:
                p = self._eval_product(t, (k, a, n))
                if p is not None:
                    exclude.append(p)
                else:
                    include.append(t)
            if not exclude:
                return None
            else:
                arg = term._new_rawargs(*include)
                A = Mul(*exclude)
                B = self.func(arg, (k, a, n)).doit()
                return without_k ** (n - a + 1) * A * B
        else:
            p = self._eval_product(with_k[0], (k, a, n))
            if p is None:
                p = self.func(with_k[0], (k, a, n)).doit()
            return without_k ** (n - a + 1) * p
    elif term.is_Pow:
        if not term.base.has(k):
            s = summation(term.exp, (k, a, n))
            return term.base ** s
        elif not term.exp.has(k):
            p = self._eval_product(term.base, (k, a, n))
            if p is not None:
                return p ** term.exp
    elif isinstance(term, Product):
        evaluated = term.doit()
        f = self._eval_product(evaluated, limits)
        if f is None:
            return self.func(evaluated, limits)
        else:
            return f
    if definite:
        return self._eval_product_direct(term, limits)