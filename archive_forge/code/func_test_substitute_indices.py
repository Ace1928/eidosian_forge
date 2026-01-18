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
def test_substitute_indices():
    Lorentz = TensorIndexType('Lorentz', dummy_name='L')
    i, j, k, l, m, n, p, q = tensor_indices('i,j,k,l,m,n,p,q', Lorentz)
    A, B = tensor_heads('A,B', [Lorentz] * 2, TensorSymmetry.fully_symmetric(2))
    p = TensorHead('p', [Lorentz])
    t = p(i)
    t1 = t.substitute_indices((j, k))
    assert t1 == t
    t1 = t.substitute_indices((i, j))
    assert t1 == p(j)
    t1 = t.substitute_indices((i, -j))
    assert t1 == p(-j)
    t1 = t.substitute_indices((-i, j))
    assert t1 == p(-j)
    t1 = t.substitute_indices((-i, -j))
    assert t1 == p(j)
    t = A(m, n)
    t1 = t.substitute_indices((m, i), (n, -i))
    assert t1 == A(n, -n)
    t1 = substitute_indices(t, (m, i), (n, -i))
    assert t1 == A(n, -n)
    t = A(i, k) * B(-k, -j)
    t1 = t.substitute_indices((i, j), (j, k))
    t1a = A(j, l) * B(-l, -k)
    assert t1 == t1a
    t1 = substitute_indices(t, (i, j), (j, k))
    assert t1 == t1a
    t = A(i, j) + B(i, j)
    t1 = t.substitute_indices((j, -i))
    t1a = A(i, -i) + B(i, -i)
    assert t1 == t1a
    t1 = substitute_indices(t, (j, -i))
    assert t1 == t1a