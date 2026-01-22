from sympy.core import (S, Add, Mul, Pow, Eq, Expr,
from sympy.core.exprtools import decompose_power, decompose_power_rat
from sympy.core.numbers import _illegal
from sympy.polys.polyerrors import PolynomialError, GeneratorsError
from sympy.polys.polyoptions import build_options
import re
class PicklableWithSlots:
    """
    Mixin class that allows to pickle objects with ``__slots__``.

    Examples
    ========

    First define a class that mixes :class:`PicklableWithSlots` in::

        >>> from sympy.polys.polyutils import PicklableWithSlots
        >>> class Some(PicklableWithSlots):
        ...     __slots__ = ('foo', 'bar')
        ...
        ...     def __init__(self, foo, bar):
        ...         self.foo = foo
        ...         self.bar = bar

    To make :mod:`pickle` happy in doctest we have to use these hacks::

        >>> import builtins
        >>> builtins.Some = Some
        >>> from sympy.polys import polyutils
        >>> polyutils.Some = Some

    Next lets see if we can create an instance, pickle it and unpickle::

        >>> some = Some('abc', 10)
        >>> some.foo, some.bar
        ('abc', 10)

        >>> from pickle import dumps, loads
        >>> some2 = loads(dumps(some))

        >>> some2.foo, some2.bar
        ('abc', 10)

    """
    __slots__ = ()

    def __getstate__(self, cls=None):
        if cls is None:
            cls = self.__class__
        d = {}
        for c in cls.__bases__:
            getstate = getattr(c, '__getstate__', None)
            objstate = getattr(object, '__getstate__', None)
            if getstate is not None and getstate is not objstate:
                d.update(getstate(self, c))
        for name in cls.__slots__:
            if hasattr(self, name):
                d[name] = getattr(self, name)
        return d

    def __setstate__(self, d):
        for name, value in d.items():
            try:
                setattr(self, name, value)
            except AttributeError:
                pass