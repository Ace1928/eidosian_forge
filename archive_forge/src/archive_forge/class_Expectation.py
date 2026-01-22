import itertools
from sympy.concrete.summations import Sum
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import expand as _expand
from sympy.core.mul import Mul
from sympy.core.relational import Eq
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.integrals.integrals import Integral
from sympy.logic.boolalg import Not
from sympy.core.parameters import global_parameters
from sympy.core.sorting import default_sort_key
from sympy.core.sympify import _sympify
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean
from sympy.stats import variance, covariance
from sympy.stats.rv import (RandomSymbol, pspace, dependent,
class Expectation(Expr):
    """
    Symbolic expression for the expectation.

    Examples
    ========

    >>> from sympy.stats import Expectation, Normal, Probability, Poisson
    >>> from sympy import symbols, Integral, Sum
    >>> mu = symbols("mu")
    >>> sigma = symbols("sigma", positive=True)
    >>> X = Normal("X", mu, sigma)
    >>> Expectation(X)
    Expectation(X)
    >>> Expectation(X).evaluate_integral().simplify()
    mu

    To get the integral expression of the expectation:

    >>> Expectation(X).rewrite(Integral)
    Integral(sqrt(2)*X*exp(-(X - mu)**2/(2*sigma**2))/(2*sqrt(pi)*sigma), (X, -oo, oo))

    The same integral expression, in more abstract terms:

    >>> Expectation(X).rewrite(Probability)
    Integral(x*Probability(Eq(X, x)), (x, -oo, oo))

    To get the Summation expression of the expectation for discrete random variables:

    >>> lamda = symbols('lamda', positive=True)
    >>> Z = Poisson('Z', lamda)
    >>> Expectation(Z).rewrite(Sum)
    Sum(Z*lamda**Z*exp(-lamda)/factorial(Z), (Z, 0, oo))

    This class is aware of some properties of the expectation:

    >>> from sympy.abc import a
    >>> Expectation(a*X)
    Expectation(a*X)
    >>> Y = Normal("Y", 1, 2)
    >>> Expectation(X + Y)
    Expectation(X + Y)

    To expand the ``Expectation`` into its expression, use ``expand()``:

    >>> Expectation(X + Y).expand()
    Expectation(X) + Expectation(Y)
    >>> Expectation(a*X + Y).expand()
    a*Expectation(X) + Expectation(Y)
    >>> Expectation(a*X + Y)
    Expectation(a*X + Y)
    >>> Expectation((X + Y)*(X - Y)).expand()
    Expectation(X**2) - Expectation(Y**2)

    To evaluate the ``Expectation``, use ``doit()``:

    >>> Expectation(X + Y).doit()
    mu + 1
    >>> Expectation(X + Expectation(Y + Expectation(2*X))).doit()
    3*mu + 1

    To prevent evaluating nested ``Expectation``, use ``doit(deep=False)``

    >>> Expectation(X + Expectation(Y)).doit(deep=False)
    mu + Expectation(Expectation(Y))
    >>> Expectation(X + Expectation(Y + Expectation(2*X))).doit(deep=False)
    mu + Expectation(Expectation(Y + Expectation(2*X)))

    """

    def __new__(cls, expr, condition=None, **kwargs):
        expr = _sympify(expr)
        if expr.is_Matrix:
            from sympy.stats.symbolic_multivariate_probability import ExpectationMatrix
            return ExpectationMatrix(expr, condition)
        if condition is None:
            if not is_random(expr):
                return expr
            obj = Expr.__new__(cls, expr)
        else:
            condition = _sympify(condition)
            obj = Expr.__new__(cls, expr, condition)
        obj._condition = condition
        return obj

    def expand(self, **hints):
        expr = self.args[0]
        condition = self._condition
        if not is_random(expr):
            return expr
        if isinstance(expr, Add):
            return Add.fromiter((Expectation(a, condition=condition).expand() for a in expr.args))
        expand_expr = _expand(expr)
        if isinstance(expand_expr, Add):
            return Add.fromiter((Expectation(a, condition=condition).expand() for a in expand_expr.args))
        elif isinstance(expr, Mul):
            rv = []
            nonrv = []
            for a in expr.args:
                if is_random(a):
                    rv.append(a)
                else:
                    nonrv.append(a)
            return Mul.fromiter(nonrv) * Expectation(Mul.fromiter(rv), condition=condition)
        return self

    def doit(self, **hints):
        deep = hints.get('deep', True)
        condition = self._condition
        expr = self.args[0]
        numsamples = hints.get('numsamples', False)
        for_rewrite = not hints.get('for_rewrite', False)
        if deep:
            expr = expr.doit(**hints)
        if not is_random(expr) or isinstance(expr, Expectation):
            return expr
        if numsamples:
            evalf = hints.get('evalf', True)
            return sampling_E(expr, condition, numsamples=numsamples, evalf=evalf)
        if expr.has(RandomIndexedSymbol):
            return pspace(expr).compute_expectation(expr, condition)
        if condition is not None:
            return self.func(given(expr, condition)).doit(**hints)
        if expr.is_Add:
            return Add(*[self.func(arg, condition).doit(**hints) if not isinstance(arg, Expectation) else self.func(arg, condition) for arg in expr.args])
        if expr.is_Mul:
            if expr.atoms(Expectation):
                return expr
        if pspace(expr) == PSpace():
            return self.func(expr)
        result = pspace(expr).compute_expectation(expr, evaluate=for_rewrite)
        if hasattr(result, 'doit') and for_rewrite:
            return result.doit(**hints)
        else:
            return result

    def _eval_rewrite_as_Probability(self, arg, condition=None, **kwargs):
        rvs = arg.atoms(RandomSymbol)
        if len(rvs) > 1:
            raise NotImplementedError()
        if len(rvs) == 0:
            return arg
        rv = rvs.pop()
        if rv.pspace is None:
            raise ValueError('Probability space not known')
        symbol = rv.symbol
        if symbol.name[0].isupper():
            symbol = Symbol(symbol.name.lower())
        else:
            symbol = Symbol(symbol.name + '_1')
        if rv.pspace.is_Continuous:
            return Integral(arg.replace(rv, symbol) * Probability(Eq(rv, symbol), condition), (symbol, rv.pspace.domain.set.inf, rv.pspace.domain.set.sup))
        elif rv.pspace.is_Finite:
            raise NotImplementedError
        else:
            return Sum(arg.replace(rv, symbol) * Probability(Eq(rv, symbol), condition), (symbol, rv.pspace.domain.set.inf, rv.pspace.set.sup))

    def _eval_rewrite_as_Integral(self, arg, condition=None, **kwargs):
        return self.func(arg, condition=condition).doit(deep=False, for_rewrite=True)
    _eval_rewrite_as_Sum = _eval_rewrite_as_Integral

    def evaluate_integral(self):
        return self.rewrite(Integral).doit()
    evaluate_sum = evaluate_integral