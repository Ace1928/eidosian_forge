import functools, itertools
from sympy.core.sympify import _sympify, sympify
from sympy.core.expr import Expr
from sympy.core import Basic, Tuple
from sympy.tensor.array import ImmutableDenseNDimArray
from sympy.core.symbol import Symbol
from sympy.core.numbers import Integer
@classmethod
def _check_limits_validity(cls, function, limits):
    new_limits = []
    for var, inf, sup in limits:
        var = _sympify(var)
        inf = _sympify(inf)
        if isinstance(sup, list):
            sup = Tuple(*sup)
        else:
            sup = _sympify(sup)
        new_limits.append(Tuple(var, inf, sup))
        if any((not isinstance(i, Expr) or i.atoms(Symbol, Integer) != i.atoms() for i in [inf, sup])):
            raise TypeError('Bounds should be an Expression(combination of Integer and Symbol)')
        if (inf > sup) == True:
            raise ValueError('Lower bound should be inferior to upper bound')
        if var in inf.free_symbols or var in sup.free_symbols:
            raise ValueError('Variable should not be part of its bounds')
    return new_limits