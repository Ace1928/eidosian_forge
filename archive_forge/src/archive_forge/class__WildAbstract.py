import re
from typing import List, Callable
from sympy.core.sympify import _sympify
from sympy.external import import_module
from sympy.functions import (log, sin, cos, tan, cot, csc, sec, erf, gamma, uppergamma)
from sympy.functions.elementary.hyperbolic import acosh, asinh, atanh, acoth, acsch, asech, cosh, sinh, tanh, coth, sech, csch
from sympy.functions.elementary.trigonometric import atan, acsc, asin, acot, acos, asec
from sympy.functions.special.error_functions import fresnelc, fresnels, erfc, erfi, Ei
from sympy.core.add import Add
from sympy.core.basic import Basic
from sympy.core.expr import Expr
from sympy.core.mul import Mul
from sympy.core.power import Pow
from sympy.core.relational import (Equality, Unequality)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.integrals.integrals import Integral
from sympy.printing.repr import srepr
from sympy.utilities.decorator import doctest_depends_on
@doctest_depends_on(modules=('matchpy',))
class _WildAbstract(Wildcard, Symbol):
    min_length: int
    fixed_size: bool

    def __init__(self, variable_name=None, optional=None, **assumptions):
        min_length = self.min_length
        fixed_size = self.fixed_size
        if optional is not None:
            optional = _sympify(optional)
        Wildcard.__init__(self, min_length, fixed_size, str(variable_name), optional)

    def __getstate__(self):
        return {'min_length': self.min_length, 'fixed_size': self.fixed_size, 'min_count': self.min_count, 'variable_name': self.variable_name, 'optional': self.optional}

    def __new__(cls, variable_name=None, optional=None, **assumptions):
        cls._sanitize(assumptions, cls)
        return _WildAbstract.__xnew__(cls, variable_name, optional, **assumptions)

    def __getnewargs__(self):
        return (self.variable_name, self.optional)

    @staticmethod
    def __xnew__(cls, variable_name=None, optional=None, **assumptions):
        obj = Symbol.__xnew__(cls, variable_name, **assumptions)
        return obj

    def _hashable_content(self):
        if self.optional:
            return super()._hashable_content() + (self.min_count, self.fixed_size, self.variable_name, self.optional)
        else:
            return super()._hashable_content() + (self.min_count, self.fixed_size, self.variable_name)

    def __copy__(self) -> '_WildAbstract':
        return type(self)(variable_name=self.variable_name, optional=self.optional)

    def __repr__(self):
        return str(self)

    def __str__(self):
        return self.name