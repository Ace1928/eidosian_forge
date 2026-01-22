from collections import defaultdict
from sympy.core.numbers import (nan, oo, zoo)
from sympy.core.add import Add
from sympy.core.expr import Expr
from sympy.core.function import Derivative, Function, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Rational
from sympy.core.relational import Eq
from sympy.sets.sets import Interval
from sympy.core.singleton import S
from sympy.core.symbol import Wild, Dummy, symbols, Symbol
from sympy.core.sympify import sympify
from sympy.discrete.convolutions import convolution
from sympy.functions.combinatorial.factorials import binomial, factorial, rf
from sympy.functions.combinatorial.numbers import bell
from sympy.functions.elementary.integers import floor, frac, ceiling
from sympy.functions.elementary.miscellaneous import Min, Max
from sympy.functions.elementary.piecewise import Piecewise
from sympy.series.limits import Limit
from sympy.series.order import Order
from sympy.series.sequences import sequence
from sympy.series.series_class import SeriesBase
from sympy.utilities.iterables import iterable
class FormalPowerSeries(SeriesBase):
    """
    Represents Formal Power Series of a function.

    Explanation
    ===========

    No computation is performed. This class should only to be used to represent
    a series. No checks are performed.

    For computing a series use :func:`fps`.

    See Also
    ========

    sympy.series.formal.fps
    """

    def __new__(cls, *args):
        args = map(sympify, args)
        return Expr.__new__(cls, *args)

    def __init__(self, *args):
        ak = args[4][0]
        k = ak.variables[0]
        self.ak_seq = sequence(ak.formula, (k, 1, oo))
        self.fact_seq = sequence(factorial(k), (k, 1, oo))
        self.bell_coeff_seq = self.ak_seq * self.fact_seq
        self.sign_seq = sequence((-1, 1), (k, 1, oo))

    @property
    def function(self):
        return self.args[0]

    @property
    def x(self):
        return self.args[1]

    @property
    def x0(self):
        return self.args[2]

    @property
    def dir(self):
        return self.args[3]

    @property
    def ak(self):
        return self.args[4][0]

    @property
    def xk(self):
        return self.args[4][1]

    @property
    def ind(self):
        return self.args[4][2]

    @property
    def interval(self):
        return Interval(0, oo)

    @property
    def start(self):
        return self.interval.inf

    @property
    def stop(self):
        return self.interval.sup

    @property
    def length(self):
        return oo

    @property
    def infinite(self):
        """Returns an infinite representation of the series"""
        from sympy.concrete import Sum
        ak, xk = (self.ak, self.xk)
        k = ak.variables[0]
        inf_sum = Sum(ak.formula * xk.formula, (k, ak.start, ak.stop))
        return self.ind + inf_sum

    def _get_pow_x(self, term):
        """Returns the power of x in a term."""
        xterm, pow_x = term.as_independent(self.x)[1].as_base_exp()
        if not xterm.has(self.x):
            return S.Zero
        return pow_x

    def polynomial(self, n=6):
        """
        Truncated series as polynomial.

        Explanation
        ===========

        Returns series expansion of ``f`` upto order ``O(x**n)``
        as a polynomial(without ``O`` term).
        """
        terms = []
        sym = self.free_symbols
        for i, t in enumerate(self):
            xp = self._get_pow_x(t)
            if xp.has(*sym):
                xp = xp.as_coeff_add(*sym)[0]
            if xp >= n:
                break
            elif xp.is_integer is True and i == n + 1:
                break
            elif t is not S.Zero:
                terms.append(t)
        return Add(*terms)

    def truncate(self, n=6):
        """
        Truncated series.

        Explanation
        ===========

        Returns truncated series expansion of f upto
        order ``O(x**n)``.

        If n is ``None``, returns an infinite iterator.
        """
        if n is None:
            return iter(self)
        x, x0 = (self.x, self.x0)
        pt_xk = self.xk.coeff(n)
        if x0 is S.NegativeInfinity:
            x0 = S.Infinity
        return self.polynomial(n) + Order(pt_xk, (x, x0))

    def zero_coeff(self):
        return self._eval_term(0)

    def _eval_term(self, pt):
        try:
            pt_xk = self.xk.coeff(pt)
            pt_ak = self.ak.coeff(pt).simplify()
        except IndexError:
            term = S.Zero
        else:
            term = pt_ak * pt_xk
        if self.ind:
            ind = S.Zero
            sym = self.free_symbols
            for t in Add.make_args(self.ind):
                pow_x = self._get_pow_x(t)
                if pow_x.has(*sym):
                    pow_x = pow_x.as_coeff_add(*sym)[0]
                if pt == 0 and pow_x < 1:
                    ind += t
                elif pow_x >= pt and pow_x < pt + 1:
                    ind += t
            term += ind
        return term.collect(self.x)

    def _eval_subs(self, old, new):
        x = self.x
        if old.has(x):
            return self

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        for t in self:
            if t is not S.Zero:
                return t

    def _eval_derivative(self, x):
        f = self.function.diff(x)
        ind = self.ind.diff(x)
        pow_xk = self._get_pow_x(self.xk.formula)
        ak = self.ak
        k = ak.variables[0]
        if ak.formula.has(x):
            form = []
            for e, c in ak.formula.args:
                temp = S.Zero
                for t in Add.make_args(e):
                    pow_x = self._get_pow_x(t)
                    temp += t * (pow_xk + pow_x)
                form.append((temp, c))
            form = Piecewise(*form)
            ak = sequence(form.subs(k, k + 1), (k, ak.start - 1, ak.stop))
        else:
            ak = sequence((ak.formula * pow_xk).subs(k, k + 1), (k, ak.start - 1, ak.stop))
        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def integrate(self, x=None, **kwargs):
        """
        Integrate Formal Power Series.

        Examples
        ========

        >>> from sympy import fps, sin, integrate
        >>> from sympy.abc import x
        >>> f = fps(sin(x))
        >>> f.integrate(x).truncate()
        -1 + x**2/2 - x**4/24 + O(x**6)
        >>> integrate(f, (x, 0, 1))
        1 - cos(1)
        """
        from sympy.integrals import integrate
        if x is None:
            x = self.x
        elif iterable(x):
            return integrate(self.function, x)
        f = integrate(self.function, x)
        ind = integrate(self.ind, x)
        ind += (f - ind).limit(x, 0)
        pow_xk = self._get_pow_x(self.xk.formula)
        ak = self.ak
        k = ak.variables[0]
        if ak.formula.has(x):
            form = []
            for e, c in ak.formula.args:
                temp = S.Zero
                for t in Add.make_args(e):
                    pow_x = self._get_pow_x(t)
                    temp += t / (pow_xk + pow_x + 1)
                form.append((temp, c))
            form = Piecewise(*form)
            ak = sequence(form.subs(k, k - 1), (k, ak.start + 1, ak.stop))
        else:
            ak = sequence((ak.formula / (pow_xk + 1)).subs(k, k - 1), (k, ak.start + 1, ak.stop))
        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def product(self, other, x=None, n=6):
        """
        Multiplies two Formal Power Series, using discrete convolution and
        return the truncated terms upto specified order.

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(sin(x))
        >>> f2 = fps(exp(x))

        >>> f1.product(f2, x).truncate(4)
        x + x**2 + x**3/3 + O(x**4)

        See Also
        ========

        sympy.discrete.convolutions
        sympy.series.formal.FormalPowerSeriesProduct

        """
        if n is None:
            return iter(self)
        other = sympify(other)
        if not isinstance(other, FormalPowerSeries):
            raise ValueError('Both series should be an instance of FormalPowerSeries class.')
        if self.dir != other.dir:
            raise ValueError('Both series should be calculated from the same direction.')
        elif self.x0 != other.x0:
            raise ValueError('Both series should be calculated about the same point.')
        elif self.x != other.x:
            raise ValueError('Both series should have the same symbol.')
        return FormalPowerSeriesProduct(self, other)

    def coeff_bell(self, n):
        """
        self.coeff_bell(n) returns a sequence of Bell polynomials of the second kind.
        Note that ``n`` should be a integer.

        The second kind of Bell polynomials (are sometimes called "partial" Bell
        polynomials or incomplete Bell polynomials) are defined as

        .. math::
            B_{n,k}(x_1, x_2,\\dotsc x_{n-k+1}) =
                \\sum_{j_1+j_2+j_2+\\dotsb=k \\atop j_1+2j_2+3j_2+\\dotsb=n}
                \\frac{n!}{j_1!j_2!\\dotsb j_{n-k+1}!}
                \\left(\\frac{x_1}{1!} \\right)^{j_1}
                \\left(\\frac{x_2}{2!} \\right)^{j_2} \\dotsb
                \\left(\\frac{x_{n-k+1}}{(n-k+1)!} \\right) ^{j_{n-k+1}}.

        * ``bell(n, k, (x1, x2, ...))`` gives Bell polynomials of the second kind,
          `B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1})`.

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell

        """
        inner_coeffs = [bell(n, j, tuple(self.bell_coeff_seq[:n - j + 1])) for j in range(1, n + 1)]
        k = Dummy('k')
        return sequence(tuple(inner_coeffs), (k, 1, oo))

    def compose(self, other, x=None, n=6):
        """
        Returns the truncated terms of the formal power series of the composed function,
        up to specified ``n``.

        Explanation
        ===========

        If ``f`` and ``g`` are two formal power series of two different functions,
        then the coefficient sequence ``ak`` of the composed formal power series `fp`
        will be as follows.

        .. math::
            \\sum\\limits_{k=0}^{n} b_k B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1})

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, sin, exp
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(sin(x))

        >>> f1.compose(f2, x).truncate()
        1 + x + x**2/2 - x**4/8 - x**5/15 + O(x**6)

        >>> f1.compose(f2, x).truncate(8)
        1 + x + x**2/2 - x**4/8 - x**5/15 - x**6/240 + x**7/90 + O(x**8)

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell
        sympy.series.formal.FormalPowerSeriesCompose

        References
        ==========

        .. [1] Comtet, Louis: Advanced combinatorics; the art of finite and infinite expansions. Reidel, 1974.

        """
        if n is None:
            return iter(self)
        other = sympify(other)
        if not isinstance(other, FormalPowerSeries):
            raise ValueError('Both series should be an instance of FormalPowerSeries class.')
        if self.dir != other.dir:
            raise ValueError('Both series should be calculated from the same direction.')
        elif self.x0 != other.x0:
            raise ValueError('Both series should be calculated about the same point.')
        elif self.x != other.x:
            raise ValueError('Both series should have the same symbol.')
        if other._eval_term(0).as_coeff_mul(other.x)[0] is not S.Zero:
            raise ValueError('The formal power series of the inner function should not have any constant coefficient term.')
        return FormalPowerSeriesCompose(self, other)

    def inverse(self, x=None, n=6):
        """
        Returns the truncated terms of the inverse of the formal power series,
        up to specified ``n``.

        Explanation
        ===========

        If ``f`` and ``g`` are two formal power series of two different functions,
        then the coefficient sequence ``ak`` of the composed formal power series ``fp``
        will be as follows.

        .. math::
            \\sum\\limits_{k=0}^{n} (-1)^{k} x_0^{-k-1} B_{n,k}(x_1, x_2, \\dotsc, x_{n-k+1})

        Parameters
        ==========

        n : Number, optional
            Specifies the order of the term up to which the polynomial should
            be truncated.

        Examples
        ========

        >>> from sympy import fps, exp, cos
        >>> from sympy.abc import x
        >>> f1 = fps(exp(x))
        >>> f2 = fps(cos(x))

        >>> f1.inverse(x).truncate()
        1 - x + x**2/2 - x**3/6 + x**4/24 - x**5/120 + O(x**6)

        >>> f2.inverse(x).truncate(8)
        1 + x**2/2 + 5*x**4/24 + 61*x**6/720 + O(x**8)

        See Also
        ========

        sympy.functions.combinatorial.numbers.bell
        sympy.series.formal.FormalPowerSeriesInverse

        References
        ==========

        .. [1] Comtet, Louis: Advanced combinatorics; the art of finite and infinite expansions. Reidel, 1974.

        """
        if n is None:
            return iter(self)
        if self._eval_term(0).is_zero:
            raise ValueError('Constant coefficient should exist for an inverse of a formal power series to exist.')
        return FormalPowerSeriesInverse(self)

    def __add__(self, other):
        other = sympify(other)
        if isinstance(other, FormalPowerSeries):
            if self.dir != other.dir:
                raise ValueError('Both series should be calculated from the same direction.')
            elif self.x0 != other.x0:
                raise ValueError('Both series should be calculated about the same point.')
            x, y = (self.x, other.x)
            f = self.function + other.function.subs(y, x)
            if self.x not in f.free_symbols:
                return f
            ak = self.ak + other.ak
            if self.ak.start > other.ak.start:
                seq = other.ak
                s, e = (other.ak.start, self.ak.start)
            else:
                seq = self.ak
                s, e = (self.ak.start, other.ak.start)
            save = Add(*[z[0] * z[1] for z in zip(seq[0:e - s], self.xk[s:e])])
            ind = self.ind + other.ind + save
            return self.func(f, x, self.x0, self.dir, (ak, self.xk, ind))
        elif not other.has(self.x):
            f = self.function + other
            ind = self.ind + other
            return self.func(f, self.x, self.x0, self.dir, (self.ak, self.xk, ind))
        return Add(self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __neg__(self):
        return self.func(-self.function, self.x, self.x0, self.dir, (-self.ak, self.xk, -self.ind))

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        return (-self).__add__(other)

    def __mul__(self, other):
        other = sympify(other)
        if other.has(self.x):
            return Mul(self, other)
        f = self.function * other
        ak = self.ak.coeff_mul(other)
        ind = self.ind * other
        return self.func(f, self.x, self.x0, self.dir, (ak, self.xk, ind))

    def __rmul__(self, other):
        return self.__mul__(other)