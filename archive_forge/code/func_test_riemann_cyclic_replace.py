from sympy.concrete.summations import Sum
from sympy.core.function import expand
from sympy.core.numbers import Integer
from sympy.matrices.dense import (Matrix, eye)
from sympy.tensor.indexed import Indexed
from sympy.combinatorics import Permutation
from sympy.core import S, Rational, Symbol, Basic, Add
from sympy.core.containers import Tuple
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.tensor.array import Array
from sympy.tensor.tensor import TensorIndexType, tensor_indices, TensorSymmetry, \
from sympy.testing.pytest import raises, XFAIL, warns_deprecated_sympy
from sympy.matrices import diag
def test_riemann_cyclic_replace():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    m0, m1, m2, m3 = tensor_indices('m:4', Lorentz)
    R = TensorHead('R', [Lorentz] * 4, TensorSymmetry.riemann())
    t = R(m0, m2, m1, m3)
    t1 = riemann_cyclic_replace(t)
    t1a = Rational(-1, 3) * R(m0, m3, m2, m1) + Rational(1, 3) * R(m0, m1, m2, m3) + Rational(2, 3) * R(m0, m2, m1, m3)
    assert t1 == t1a