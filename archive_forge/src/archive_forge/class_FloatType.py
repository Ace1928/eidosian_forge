from __future__ import annotations
from typing import Any
from collections import defaultdict
from sympy.core.relational import (Ge, Gt, Le, Lt)
from sympy.core import Symbol, Tuple, Dummy
from sympy.core.basic import Basic
from sympy.core.expr import Expr, Atom
from sympy.core.numbers import Float, Integer, oo
from sympy.core.sympify import _sympify, sympify, SympifyError
from sympy.utilities.iterables import (iterable, topological_sort,
class FloatType(FloatBaseType):
    """ Represents a floating point type with fixed bit width.

    Base 2 & one sign bit is assumed.

    Parameters
    ==========

    name : str
        Name of the type.
    nbits : integer
        Number of bits used (storage).
    nmant : integer
        Number of bits used to represent the mantissa.
    nexp : integer
        Number of bits used to represent the mantissa.

    Examples
    ========

    >>> from sympy import S
    >>> from sympy.codegen.ast import FloatType
    >>> half_precision = FloatType('f16', nbits=16, nmant=10, nexp=5)
    >>> half_precision.max
    65504
    >>> half_precision.tiny == S(2)**-14
    True
    >>> half_precision.eps == S(2)**-10
    True
    >>> half_precision.dig == 3
    True
    >>> half_precision.decimal_dig == 5
    True
    >>> half_precision.cast_check(1.0)
    1.0
    >>> half_precision.cast_check(1e5)  # doctest: +ELLIPSIS
    Traceback (most recent call last):
      ...
    ValueError: Maximum value for data type smaller than new value.
    """
    __slots__ = ('nbits', 'nmant', 'nexp')
    _fields = Type._fields + __slots__
    _construct_nbits = _construct_nmant = _construct_nexp = Integer

    @property
    def max_exponent(self):
        """ The largest positive number n, such that 2**(n - 1) is a representable finite value. """
        return two ** (self.nexp - 1)

    @property
    def min_exponent(self):
        """ The lowest negative number n, such that 2**(n - 1) is a valid normalized number. """
        return 3 - self.max_exponent

    @property
    def max(self):
        """ Maximum value representable. """
        return (1 - two ** (-(self.nmant + 1))) * two ** self.max_exponent

    @property
    def tiny(self):
        """ The minimum positive normalized value. """
        return two ** (self.min_exponent - 1)

    @property
    def eps(self):
        """ Difference between 1.0 and the next representable value. """
        return two ** (-self.nmant)

    @property
    def dig(self):
        """ Number of decimal digits that are guaranteed to be preserved in text.

        When converting text -> float -> text, you are guaranteed that at least ``dig``
        number of digits are preserved with respect to rounding or overflow.
        """
        from sympy.functions import floor, log
        return floor(self.nmant * log(2) / log(10))

    @property
    def decimal_dig(self):
        """ Number of digits needed to store & load without loss.

        Explanation
        ===========

        Number of decimal digits needed to guarantee that two consecutive conversions
        (float -> text -> float) to be idempotent. This is useful when one do not want
        to loose precision due to rounding errors when storing a floating point value
        as text.
        """
        from sympy.functions import ceiling, log
        return ceiling((self.nmant + 1) * log(2) / log(10) + 1)

    def cast_nocheck(self, value):
        """ Casts without checking if out of bounds or subnormal. """
        if value == oo:
            return float(oo)
        elif value == -oo:
            return float(-oo)
        return Float(str(sympify(value).evalf(self.decimal_dig)), self.decimal_dig)

    def _check(self, value):
        if value < -self.max:
            raise ValueError('Value is too small: %d < %d' % (value, -self.max))
        if value > self.max:
            raise ValueError('Value is too big: %d > %d' % (value, self.max))
        if abs(value) < self.tiny:
            raise ValueError('Smallest (absolute) value for data type bigger than new value.')