from sympy.core.add import Add
from sympy.core.containers import Tuple
from sympy.core.function import (Function, Lambda)
from sympy.core.mul import Mul
from sympy.core.numbers import (Float, I, Integer, Rational, pi, oo)
from sympy.core.power import Pow
from sympy.core.singleton import S
from sympy.core.symbol import Symbol
from sympy.functions.elementary.complexes import Abs
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.functions.elementary.trigonometric import (cos, sin)
from sympy.logic.boolalg import (false, Or, true, Xor)
from sympy.matrices.dense import Matrix
from sympy.parsing.sympy_parser import null
from sympy.polys.polytools import Poly
from sympy.printing.repr import srepr
from sympy.sets.fancysets import Range
from sympy.sets.sets import Interval
from sympy.abc import x, y
from sympy.core.sympify import (sympify, _sympify, SympifyError, kernS,
from sympy.core.decorators import _sympifyit
from sympy.external import import_module
from sympy.testing.pytest import raises, XFAIL, skip, warns_deprecated_sympy
from sympy.utilities.decorator import conserve_mpmath_dps
from sympy.geometry import Point, Line
from sympy.functions.combinatorial.factorials import factorial, factorial2
from sympy.abc import _clash, _clash1, _clash2
from sympy.external.gmpy import HAS_GMPY
from sympy.sets import FiniteSet, EmptySet
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
import mpmath
from collections import defaultdict, OrderedDict
from mpmath.rational import mpq
def test_numpy_sympify_args():
    if not numpy:
        skip('numpy not installed.')
    a = sympify(numpy.str_('a'))
    assert type(a) is Symbol
    assert a == Symbol('a')

    class CustomSymbol(Symbol):
        pass
    a = sympify(numpy.str_('a'), {'Symbol': CustomSymbol})
    assert isinstance(a, CustomSymbol)
    a = sympify(numpy.str_('x^y'))
    assert a == x ** y
    a = sympify(numpy.str_('x^y'), convert_xor=False)
    assert a == Xor(x, y)
    raises(SympifyError, lambda: sympify(numpy.str_('x'), strict=True))
    a = sympify(numpy.str_('1.1'))
    assert isinstance(a, Float)
    assert a == 1.1
    a = sympify(numpy.str_('1.1'), rational=True)
    assert isinstance(a, Rational)
    assert a == Rational(11, 10)
    a = sympify(numpy.str_('x + x'))
    assert isinstance(a, Mul)
    assert a == 2 * x
    a = sympify(numpy.str_('x + x'), evaluate=False)
    assert isinstance(a, Add)
    assert a == Add(x, x, evaluate=False)