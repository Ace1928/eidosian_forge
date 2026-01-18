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
def test_sympify_numpy():
    if not numpy:
        skip('numpy not installed. Abort numpy tests.')
    np = numpy

    def equal(x, y):
        return x == y and type(x) == type(y)
    assert sympify(np.bool_(1)) is S(True)
    try:
        assert equal(sympify(np.int_(1234567891234567891)), S(1234567891234567891))
        assert equal(sympify(np.intp(1234567891234567891)), S(1234567891234567891))
    except OverflowError:
        pass
    assert equal(sympify(np.intc(1234567891)), S(1234567891))
    assert equal(sympify(np.int8(-123)), S(-123))
    assert equal(sympify(np.int16(-12345)), S(-12345))
    assert equal(sympify(np.int32(-1234567891)), S(-1234567891))
    assert equal(sympify(np.int64(-1234567891234567891)), S(-1234567891234567891))
    assert equal(sympify(np.uint8(123)), S(123))
    assert equal(sympify(np.uint16(12345)), S(12345))
    assert equal(sympify(np.uint32(1234567891)), S(1234567891))
    assert equal(sympify(np.uint64(1234567891234567891)), S(1234567891234567891))
    assert equal(sympify(np.float32(1.123456)), Float(1.123456, precision=24))
    assert equal(sympify(np.float64(1.1234567891234)), Float(1.1234567891234, precision=53))
    ldprec = np.finfo(np.longdouble(1)).nmant + 1
    assert equal(sympify(np.longdouble(1.123456789)), Float(1.123456789, precision=ldprec))
    assert equal(sympify(np.complex64(1 + 2j)), S(1.0 + 2.0 * I))
    assert equal(sympify(np.complex128(1 + 2j)), S(1.0 + 2.0 * I))
    lcprec = np.finfo(np.longcomplex(1)).nmant + 1
    assert equal(sympify(np.longcomplex(1 + 2j)), Float(1.0, precision=lcprec) + Float(2.0, precision=lcprec) * I)
    if hasattr(np, 'float96'):
        f96prec = np.finfo(np.float96(1)).nmant + 1
        assert equal(sympify(np.float96(1.123456789)), Float(1.123456789, precision=f96prec))
    if hasattr(np, 'float128'):
        f128prec = np.finfo(np.float128(1)).nmant + 1
        assert equal(sympify(np.float128(1.123456789123)), Float(1.123456789123, precision=f128prec))