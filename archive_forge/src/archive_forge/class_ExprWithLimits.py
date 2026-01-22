from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.expr import Expr
from sympy.core.function import AppliedUndef, UndefinedFunction
from sympy.core.mul import Mul
from sympy.core.relational import Equality, Relational
from sympy.core.singleton import S
from sympy.core.symbol import Symbol, Dummy
from sympy.core.sympify import sympify
from sympy.functions.elementary.piecewise import (piecewise_fold,
from sympy.logic.boolalg import BooleanFunction
from sympy.matrices.matrices import MatrixBase
from sympy.sets.sets import Interval, Set
from sympy.sets.fancysets import Range
from sympy.tensor.indexed import Idx
from sympy.utilities import flatten
from sympy.utilities.iterables import sift, is_sequence
from sympy.utilities.exceptions import sympy_deprecation_warning
class ExprWithLimits(Expr):
    __slots__ = ('is_commutative',)

    def __new__(cls, function, *symbols, **assumptions):
        from sympy.concrete.products import Product
        pre = _common_new(cls, function, *symbols, discrete=issubclass(cls, Product), **assumptions)
        if isinstance(pre, tuple):
            function, limits, _ = pre
        else:
            return pre
        if any((len(l) != 3 or None in l for l in limits)):
            raise ValueError('ExprWithLimits requires values for lower and upper bounds.')
        obj = Expr.__new__(cls, **assumptions)
        arglist = [function]
        arglist.extend(limits)
        obj._args = tuple(arglist)
        obj.is_commutative = function.is_commutative
        return obj

    @property
    def function(self):
        """Return the function applied across limits.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x
        >>> Integral(x**2, (x,)).function
        x**2

        See Also
        ========

        limits, variables, free_symbols
        """
        return self._args[0]

    @property
    def kind(self):
        return self.function.kind

    @property
    def limits(self):
        """Return the limits of expression.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, i
        >>> Integral(x**i, (i, 1, 3)).limits
        ((i, 1, 3),)

        See Also
        ========

        function, variables, free_symbols
        """
        return self._args[1:]

    @property
    def variables(self):
        """Return a list of the limit variables.

        >>> from sympy import Sum
        >>> from sympy.abc import x, i
        >>> Sum(x**i, (i, 1, 3)).variables
        [i]

        See Also
        ========

        function, limits, free_symbols
        as_dummy : Rename dummy variables
        sympy.integrals.integrals.Integral.transform : Perform mapping on the dummy variable
        """
        return [l[0] for l in self.limits]

    @property
    def bound_symbols(self):
        """Return only variables that are dummy variables.

        Examples
        ========

        >>> from sympy import Integral
        >>> from sympy.abc import x, i, j, k
        >>> Integral(x**i, (i, 1, 3), (j, 2), k).bound_symbols
        [i, j]

        See Also
        ========

        function, limits, free_symbols
        as_dummy : Rename dummy variables
        sympy.integrals.integrals.Integral.transform : Perform mapping on the dummy variable
        """
        return [l[0] for l in self.limits if len(l) != 1]

    @property
    def free_symbols(self):
        """
        This method returns the symbols in the object, excluding those
        that take on a specific value (i.e. the dummy symbols).

        Examples
        ========

        >>> from sympy import Sum
        >>> from sympy.abc import x, y
        >>> Sum(x, (x, y, 1)).free_symbols
        {y}
        """
        function, limits = (self.function, self.limits)
        reps = {i[0]: i[0] if i[0].free_symbols == {i[0]} else Dummy() for i in self.limits}
        function = function.xreplace(reps)
        isyms = function.free_symbols
        for xab in limits:
            v = reps[xab[0]]
            if len(xab) == 1:
                isyms.add(v)
                continue
            if v in isyms:
                isyms.remove(v)
            for i in xab[1:]:
                isyms.update(i.free_symbols)
        reps = {v: k for k, v in reps.items()}
        return {reps.get(_, _) for _ in isyms}

    @property
    def is_number(self):
        """Return True if the Sum has no free symbols, else False."""
        return not self.free_symbols

    def _eval_interval(self, x, a, b):
        limits = [i if i[0] != x else (x, a, b) for i in self.limits]
        integrand = self.function
        return self.func(integrand, *limits)

    def _eval_subs(self, old, new):
        """
        Perform substitutions over non-dummy variables
        of an expression with limits.  Also, can be used
        to specify point-evaluation of an abstract antiderivative.

        Examples
        ========

        >>> from sympy import Sum, oo
        >>> from sympy.abc import s, n
        >>> Sum(1/n**s, (n, 1, oo)).subs(s, 2)
        Sum(n**(-2), (n, 1, oo))

        >>> from sympy import Integral
        >>> from sympy.abc import x, a
        >>> Integral(a*x**2, x).subs(x, 4)
        Integral(a*x**2, (x, 4))

        See Also
        ========

        variables : Lists the integration variables
        transform : Perform mapping on the dummy variable for integrals
        change_index : Perform mapping on the sum and product dummy variables

        """
        func, limits = (self.function, list(self.limits))
        limits.reverse()
        if not isinstance(old, Symbol) or old.free_symbols.intersection(self.free_symbols):
            sub_into_func = True
            for i, xab in enumerate(limits):
                if 1 == len(xab) and old == xab[0]:
                    if new._diff_wrt:
                        xab = (new,)
                    else:
                        xab = (old, old)
                limits[i] = Tuple(xab[0], *[l._subs(old, new) for l in xab[1:]])
                if len(xab[0].free_symbols.intersection(old.free_symbols)) != 0:
                    sub_into_func = False
                    break
            if isinstance(old, (AppliedUndef, UndefinedFunction)):
                sy2 = set(self.variables).intersection(set(new.atoms(Symbol)))
                sy1 = set(self.variables).intersection(set(old.args))
                if not sy2.issubset(sy1):
                    raise ValueError('substitution cannot create dummy dependencies')
                sub_into_func = True
            if sub_into_func:
                func = func.subs(old, new)
        else:
            for i, xab in enumerate(limits):
                if len(xab) == 3:
                    limits[i] = Tuple(xab[0], *[l._subs(old, new) for l in xab[1:]])
                    if old == xab[0]:
                        break
        for i, xab in enumerate(limits):
            if len(xab) == 2 and (xab[0] - xab[1]).is_zero:
                limits[i] = Tuple(xab[0])
        limits.reverse()
        return self.func(func, *limits)

    @property
    def has_finite_limits(self):
        """
        Returns True if the limits are known to be finite, either by the
        explicit bounds, assumptions on the bounds, or assumptions on the
        variables.  False if known to be infinite, based on the bounds.
        None if not enough information is available to determine.

        Examples
        ========

        >>> from sympy import Sum, Integral, Product, oo, Symbol
        >>> x = Symbol('x')
        >>> Sum(x, (x, 1, 8)).has_finite_limits
        True

        >>> Integral(x, (x, 1, oo)).has_finite_limits
        False

        >>> M = Symbol('M')
        >>> Sum(x, (x, 1, M)).has_finite_limits

        >>> N = Symbol('N', integer=True)
        >>> Product(x, (x, 1, N)).has_finite_limits
        True

        See Also
        ========

        has_reversed_limits

        """
        ret_None = False
        for lim in self.limits:
            if len(lim) == 3:
                if any((l.is_infinite for l in lim[1:])):
                    return False
                elif any((l.is_infinite is None for l in lim[1:])):
                    if lim[0].is_infinite is None:
                        ret_None = True
            elif lim[0].is_infinite is None:
                ret_None = True
        if ret_None:
            return None
        return True

    @property
    def has_reversed_limits(self):
        """
        Returns True if the limits are known to be in reversed order, either
        by the explicit bounds, assumptions on the bounds, or assumptions on the
        variables.  False if known to be in normal order, based on the bounds.
        None if not enough information is available to determine.

        Examples
        ========

        >>> from sympy import Sum, Integral, Product, oo, Symbol
        >>> x = Symbol('x')
        >>> Sum(x, (x, 8, 1)).has_reversed_limits
        True

        >>> Sum(x, (x, 1, oo)).has_reversed_limits
        False

        >>> M = Symbol('M')
        >>> Integral(x, (x, 1, M)).has_reversed_limits

        >>> N = Symbol('N', integer=True, positive=True)
        >>> Sum(x, (x, 1, N)).has_reversed_limits
        False

        >>> Product(x, (x, 2, N)).has_reversed_limits

        >>> Product(x, (x, 2, N)).subs(N, N + 2).has_reversed_limits
        False

        See Also
        ========

        sympy.concrete.expr_with_intlimits.ExprWithIntLimits.has_empty_sequence

        """
        ret_None = False
        for lim in self.limits:
            if len(lim) == 3:
                var, a, b = lim
                dif = b - a
                if dif.is_extended_negative:
                    return True
                elif dif.is_extended_nonnegative:
                    continue
                else:
                    ret_None = True
            else:
                return None
        if ret_None:
            return None
        return False