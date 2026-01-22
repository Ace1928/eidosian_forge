from sympy.core.basic import Basic
from sympy.core import (S, Expr, Integer, Float, I, oo, Add, Lambda,
from sympy.core.cache import cacheit
from sympy.core.relational import is_le
from sympy.core.sorting import ordered
from sympy.polys.domains import QQ
from sympy.polys.polyerrors import (
from sympy.polys.polyfuncs import symmetrize, viete
from sympy.polys.polyroots import (
from sympy.polys.polytools import Poly, PurePoly, factor
from sympy.polys.rationaltools import together
from sympy.polys.rootisolation import (
from sympy.utilities import lambdify, public, sift, numbered_symbols
from mpmath import mpf, mpc, findroot, workprec
from mpmath.libmp.libmpf import dps_to_prec, prec_to_dps
from sympy.multipledispatch import dispatch
from itertools import chain
@public
class RootSum(Expr):
    """Represents a sum of all roots of a univariate polynomial. """
    __slots__ = ('poly', 'fun', 'auto')

    def __new__(cls, expr, func=None, x=None, auto=True, quadratic=False):
        """Construct a new ``RootSum`` instance of roots of a polynomial."""
        coeff, poly = cls._transform(expr, x)
        if not poly.is_univariate:
            raise MultivariatePolynomialError('only univariate polynomials are allowed')
        if func is None:
            func = Lambda(poly.gen, poly.gen)
        else:
            is_func = getattr(func, 'is_Function', False)
            if is_func and 1 in func.nargs:
                if not isinstance(func, Lambda):
                    func = Lambda(poly.gen, func(poly.gen))
            else:
                raise ValueError('expected a univariate function, got %s' % func)
        var, expr = (func.variables[0], func.expr)
        if coeff is not S.One:
            expr = expr.subs(var, coeff * var)
        deg = poly.degree()
        if not expr.has(var):
            return deg * expr
        if expr.is_Add:
            add_const, expr = expr.as_independent(var)
        else:
            add_const = S.Zero
        if expr.is_Mul:
            mul_const, expr = expr.as_independent(var)
        else:
            mul_const = S.One
        func = Lambda(var, expr)
        rational = cls._is_func_rational(poly, func)
        factors, terms = (_pure_factors(poly), [])
        for poly, k in factors:
            if poly.is_linear:
                term = func(roots_linear(poly)[0])
            elif quadratic and poly.is_quadratic:
                term = sum(map(func, roots_quadratic(poly)))
            elif not rational or not auto:
                term = cls._new(poly, func, auto)
            else:
                term = cls._rational_case(poly, func)
            terms.append(k * term)
        return mul_const * Add(*terms) + deg * add_const

    @classmethod
    def _new(cls, poly, func, auto=True):
        """Construct new raw ``RootSum`` instance. """
        obj = Expr.__new__(cls)
        obj.poly = poly
        obj.fun = func
        obj.auto = auto
        return obj

    @classmethod
    def new(cls, poly, func, auto=True):
        """Construct new ``RootSum`` instance. """
        if not func.expr.has(*func.variables):
            return func.expr
        rational = cls._is_func_rational(poly, func)
        if not rational or not auto:
            return cls._new(poly, func, auto)
        else:
            return cls._rational_case(poly, func)

    @classmethod
    def _transform(cls, expr, x):
        """Transform an expression to a polynomial. """
        poly = PurePoly(expr, x, greedy=False)
        return preprocess_roots(poly)

    @classmethod
    def _is_func_rational(cls, poly, func):
        """Check if a lambda is a rational function. """
        var, expr = (func.variables[0], func.expr)
        return expr.is_rational_function(var)

    @classmethod
    def _rational_case(cls, poly, func):
        """Handle the rational function case. """
        roots = symbols('r:%d' % poly.degree())
        var, expr = (func.variables[0], func.expr)
        f = sum((expr.subs(var, r) for r in roots))
        p, q = together(f).as_numer_denom()
        domain = QQ[roots]
        p = p.expand()
        q = q.expand()
        try:
            p = Poly(p, domain=domain, expand=False)
        except GeneratorsNeeded:
            p, p_coeff = (None, (p,))
        else:
            p_monom, p_coeff = zip(*p.terms())
        try:
            q = Poly(q, domain=domain, expand=False)
        except GeneratorsNeeded:
            q, q_coeff = (None, (q,))
        else:
            q_monom, q_coeff = zip(*q.terms())
        coeffs, mapping = symmetrize(p_coeff + q_coeff, formal=True)
        formulas, values = (viete(poly, roots), [])
        for (sym, _), (_, val) in zip(mapping, formulas):
            values.append((sym, val))
        for i, (coeff, _) in enumerate(coeffs):
            coeffs[i] = coeff.subs(values)
        n = len(p_coeff)
        p_coeff = coeffs[:n]
        q_coeff = coeffs[n:]
        if p is not None:
            p = Poly(dict(zip(p_monom, p_coeff)), *p.gens).as_expr()
        else:
            p, = p_coeff
        if q is not None:
            q = Poly(dict(zip(q_monom, q_coeff)), *q.gens).as_expr()
        else:
            q, = q_coeff
        return factor(p / q)

    def _hashable_content(self):
        return (self.poly, self.fun)

    @property
    def expr(self):
        return self.poly.as_expr()

    @property
    def args(self):
        return (self.expr, self.fun, self.poly.gen)

    @property
    def free_symbols(self):
        return self.poly.free_symbols | self.fun.free_symbols

    @property
    def is_commutative(self):
        return True

    def doit(self, **hints):
        if not hints.get('roots', True):
            return self
        _roots = roots(self.poly, multiple=True)
        if len(_roots) < self.poly.degree():
            return self
        else:
            return Add(*[self.fun(r) for r in _roots])

    def _eval_evalf(self, prec):
        try:
            _roots = self.poly.nroots(n=prec_to_dps(prec))
        except (DomainError, PolynomialError):
            return self
        else:
            return Add(*[self.fun(r) for r in _roots])

    def _eval_derivative(self, x):
        var, expr = self.fun.args
        func = Lambda(var, expr.diff(x))
        return self.new(self.poly, func, self.auto)