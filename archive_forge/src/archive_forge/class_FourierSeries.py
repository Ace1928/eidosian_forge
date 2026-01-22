from sympy.core.numbers import (oo, pi)
from sympy.core.symbol import Wild
from sympy.core.expr import Expr
from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.singleton import S
from sympy.core.symbol import Dummy, Symbol
from sympy.core.sympify import sympify
from sympy.functions.elementary.trigonometric import sin, cos, sinc
from sympy.series.series_class import SeriesBase
from sympy.series.sequences import SeqFormula
from sympy.sets.sets import Interval
from sympy.utilities.iterables import is_sequence
class FourierSeries(SeriesBase):
    """Represents Fourier sine/cosine series.

    Explanation
    ===========

    This class only represents a fourier series.
    No computation is performed.

    For how to compute Fourier series, see the :func:`fourier_series`
    docstring.

    See Also
    ========

    sympy.series.fourier.fourier_series
    """

    def __new__(cls, *args):
        args = map(sympify, args)
        return Expr.__new__(cls, *args)

    @property
    def function(self):
        return self.args[0]

    @property
    def x(self):
        return self.args[1][0]

    @property
    def period(self):
        return (self.args[1][1], self.args[1][2])

    @property
    def a0(self):
        return self.args[2][0]

    @property
    def an(self):
        return self.args[2][1]

    @property
    def bn(self):
        return self.args[2][2]

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
    def L(self):
        return abs(self.period[1] - self.period[0]) / 2

    def _eval_subs(self, old, new):
        x = self.x
        if old.has(x):
            return self

    def truncate(self, n=3):
        """
        Return the first n nonzero terms of the series.

        If ``n`` is None return an iterator.

        Parameters
        ==========

        n : int or None
            Amount of non-zero terms in approximation or None.

        Returns
        =======

        Expr or iterator :
            Approximation of function expanded into Fourier series.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x, (x, -pi, pi))
        >>> s.truncate(4)
        2*sin(x) - sin(2*x) + 2*sin(3*x)/3 - sin(4*x)/2

        See Also
        ========

        sympy.series.fourier.FourierSeries.sigma_approximation
        """
        if n is None:
            return iter(self)
        terms = []
        for t in self:
            if len(terms) == n:
                break
            if t is not S.Zero:
                terms.append(t)
        return Add(*terms)

    def sigma_approximation(self, n=3):
        """
        Return :math:`\\sigma`-approximation of Fourier series with respect
        to order n.

        Explanation
        ===========

        Sigma approximation adjusts a Fourier summation to eliminate the Gibbs
        phenomenon which would otherwise occur at discontinuities.
        A sigma-approximated summation for a Fourier series of a T-periodical
        function can be written as

        .. math::
            s(\\theta) = \\frac{1}{2} a_0 + \\sum _{k=1}^{m-1}
            \\operatorname{sinc} \\Bigl( \\frac{k}{m} \\Bigr) \\cdot
            \\left[ a_k \\cos \\Bigl( \\frac{2\\pi k}{T} \\theta \\Bigr)
            + b_k \\sin \\Bigl( \\frac{2\\pi k}{T} \\theta \\Bigr) \\right],

        where :math:`a_0, a_k, b_k, k=1,\\ldots,{m-1}` are standard Fourier
        series coefficients and
        :math:`\\operatorname{sinc} \\Bigl( \\frac{k}{m} \\Bigr)` is a Lanczos
        :math:`\\sigma` factor (expressed in terms of normalized
        :math:`\\operatorname{sinc}` function).

        Parameters
        ==========

        n : int
            Highest order of the terms taken into account in approximation.

        Returns
        =======

        Expr :
            Sigma approximation of function expanded into Fourier series.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x, (x, -pi, pi))
        >>> s.sigma_approximation(4)
        2*sin(x)*sinc(pi/4) - 2*sin(2*x)/pi + 2*sin(3*x)*sinc(3*pi/4)/3

        See Also
        ========

        sympy.series.fourier.FourierSeries.truncate

        Notes
        =====

        The behaviour of
        :meth:`~sympy.series.fourier.FourierSeries.sigma_approximation`
        is different from :meth:`~sympy.series.fourier.FourierSeries.truncate`
        - it takes all nonzero terms of degree smaller than n, rather than
        first n nonzero ones.

        References
        ==========

        .. [1] https://en.wikipedia.org/wiki/Gibbs_phenomenon
        .. [2] https://en.wikipedia.org/wiki/Sigma_approximation
        """
        terms = [sinc(pi * i / n) * t for i, t in enumerate(self[:n]) if t is not S.Zero]
        return Add(*terms)

    def shift(self, s):
        """
        Shift the function by a term independent of x.

        Explanation
        ===========

        f(x) -> f(x) + s

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.shift(1).truncate()
        -4*cos(x) + cos(2*x) + 1 + pi**2/3
        """
        s, x = (sympify(s), self.x)
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))
        a0 = self.a0 + s
        sfunc = self.function + s
        return self.func(sfunc, self.args[1], (a0, self.an, self.bn))

    def shiftx(self, s):
        """
        Shift x by a term independent of x.

        Explanation
        ===========

        f(x) -> f(x + s)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.shiftx(1).truncate()
        -4*cos(x + 1) + cos(2*x + 2) + pi**2/3
        """
        s, x = (sympify(s), self.x)
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))
        an = self.an.subs(x, x + s)
        bn = self.bn.subs(x, x + s)
        sfunc = self.function.subs(x, x + s)
        return self.func(sfunc, self.args[1], (self.a0, an, bn))

    def scale(self, s):
        """
        Scale the function by a term independent of x.

        Explanation
        ===========

        f(x) -> s * f(x)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.scale(2).truncate()
        -8*cos(x) + 2*cos(2*x) + 2*pi**2/3
        """
        s, x = (sympify(s), self.x)
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))
        an = self.an.coeff_mul(s)
        bn = self.bn.coeff_mul(s)
        a0 = self.a0 * s
        sfunc = self.args[0] * s
        return self.func(sfunc, self.args[1], (a0, an, bn))

    def scalex(self, s):
        """
        Scale x by a term independent of x.

        Explanation
        ===========

        f(x) -> f(s*x)

        This is fast, if Fourier series of f(x) is already
        computed.

        Examples
        ========

        >>> from sympy import fourier_series, pi
        >>> from sympy.abc import x
        >>> s = fourier_series(x**2, (x, -pi, pi))
        >>> s.scalex(2).truncate()
        -4*cos(2*x) + cos(4*x) + pi**2/3
        """
        s, x = (sympify(s), self.x)
        if x in s.free_symbols:
            raise ValueError("'%s' should be independent of %s" % (s, x))
        an = self.an.subs(x, x * s)
        bn = self.bn.subs(x, x * s)
        sfunc = self.function.subs(x, x * s)
        return self.func(sfunc, self.args[1], (self.a0, an, bn))

    def _eval_as_leading_term(self, x, logx=None, cdir=0):
        for t in self:
            if t is not S.Zero:
                return t

    def _eval_term(self, pt):
        if pt == 0:
            return self.a0
        return self.an.coeff(pt) + self.bn.coeff(pt)

    def __neg__(self):
        return self.scale(-1)

    def __add__(self, other):
        if isinstance(other, FourierSeries):
            if self.period != other.period:
                raise ValueError('Both the series should have same periods')
            x, y = (self.x, other.x)
            function = self.function + other.function.subs(y, x)
            if self.x not in function.free_symbols:
                return function
            an = self.an + other.an
            bn = self.bn + other.bn
            a0 = self.a0 + other.a0
            return self.func(function, self.args[1], (a0, an, bn))
        return Add(self, other)

    def __sub__(self, other):
        return self.__add__(-other)