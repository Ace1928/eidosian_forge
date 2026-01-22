from sympy.core.basic import Basic
from sympy.core.cache import cacheit
from sympy.core.containers import Tuple
from sympy.core.decorators import call_highest_priority
from sympy.core.parameters import global_parameters
from sympy.core.function import AppliedUndef, expand
from sympy.core.mul import Mul
from sympy.core.numbers import Integer
from sympy.core.relational import Eq
from sympy.core.singleton import S, Singleton
from sympy.core.sorting import ordered
from sympy.core.symbol import Dummy, Symbol, Wild
from sympy.core.sympify import sympify
from sympy.matrices import Matrix
from sympy.polys import lcm, factor
from sympy.sets.sets import Interval, Intersection
from sympy.tensor.indexed import Idx
from sympy.utilities.iterables import flatten, is_sequence, iterable
class SeqPer(SeqExpr):
    """
    Represents a periodic sequence.

    The elements are repeated after a given period.

    Examples
    ========

    >>> from sympy import SeqPer, oo
    >>> from sympy.abc import k

    >>> s = SeqPer((1, 2, 3), (0, 5))
    >>> s.periodical
    (1, 2, 3)
    >>> s.period
    3

    For value at a particular point

    >>> s.coeff(3)
    1

    supports slicing

    >>> s[:]
    [1, 2, 3, 1, 2, 3]

    iterable

    >>> list(s)
    [1, 2, 3, 1, 2, 3]

    sequence starts from negative infinity

    >>> SeqPer((1, 2, 3), (-oo, 0))[0:6]
    [1, 2, 3, 1, 2, 3]

    Periodic formulas

    >>> SeqPer((k, k**2, k**3), (k, 0, oo))[0:6]
    [0, 1, 8, 3, 16, 125]

    See Also
    ========

    sympy.series.sequences.SeqFormula
    """

    def __new__(cls, periodical, limits=None):
        periodical = sympify(periodical)

        def _find_x(periodical):
            free = periodical.free_symbols
            if len(periodical.free_symbols) == 1:
                return free.pop()
            else:
                return Dummy('k')
        x, start, stop = (None, None, None)
        if limits is None:
            x, start, stop = (_find_x(periodical), 0, S.Infinity)
        if is_sequence(limits, Tuple):
            if len(limits) == 3:
                x, start, stop = limits
            elif len(limits) == 2:
                x = _find_x(periodical)
                start, stop = limits
        if not isinstance(x, (Symbol, Idx)) or start is None or stop is None:
            raise ValueError('Invalid limits given: %s' % str(limits))
        if start is S.NegativeInfinity and stop is S.Infinity:
            raise ValueError('Both the start and end valuecannot be unbounded')
        limits = sympify((x, start, stop))
        if is_sequence(periodical, Tuple):
            periodical = sympify(tuple(flatten(periodical)))
        else:
            raise ValueError('invalid period %s should be something like e.g (1, 2) ' % periodical)
        if Interval(limits[1], limits[2]) is S.EmptySet:
            return S.EmptySequence
        return Basic.__new__(cls, periodical, limits)

    @property
    def period(self):
        return len(self.gen)

    @property
    def periodical(self):
        return self.gen

    def _eval_coeff(self, pt):
        if self.start is S.NegativeInfinity:
            idx = (self.stop - pt) % self.period
        else:
            idx = (pt - self.start) % self.period
        return self.periodical[idx].subs(self.variables[0], pt)

    def _add(self, other):
        """See docstring of SeqBase._add"""
        if isinstance(other, SeqPer):
            per1, lper1 = (self.periodical, self.period)
            per2, lper2 = (other.periodical, other.period)
            per_length = lcm(lper1, lper2)
            new_per = []
            for x in range(per_length):
                ele1 = per1[x % lper1]
                ele2 = per2[x % lper2]
                new_per.append(ele1 + ele2)
            start, stop = self._intersect_interval(other)
            return SeqPer(new_per, (self.variables[0], start, stop))

    def _mul(self, other):
        """See docstring of SeqBase._mul"""
        if isinstance(other, SeqPer):
            per1, lper1 = (self.periodical, self.period)
            per2, lper2 = (other.periodical, other.period)
            per_length = lcm(lper1, lper2)
            new_per = []
            for x in range(per_length):
                ele1 = per1[x % lper1]
                ele2 = per2[x % lper2]
                new_per.append(ele1 * ele2)
            start, stop = self._intersect_interval(other)
            return SeqPer(new_per, (self.variables[0], start, stop))

    def coeff_mul(self, coeff):
        """See docstring of SeqBase.coeff_mul"""
        coeff = sympify(coeff)
        per = [x * coeff for x in self.periodical]
        return SeqPer(per, self.args[1])