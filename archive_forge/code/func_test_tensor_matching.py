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
def test_tensor_matching():
    """
    Test match and replace with the pattern being a WildTensor or a WildTensorIndex
    """
    R3 = TensorIndexType('R3', dim=3)
    p, q, r = tensor_indices('p q r', R3)
    a, b, c = symbols('a b c', cls=WildTensorIndex, tensor_index_type=R3, ignore_updown=True)
    g = WildTensorIndex('g', R3)
    eps = R3.epsilon
    K = TensorHead('K', [R3])
    V = TensorHead('V', [R3])
    A = TensorHead('A', [R3, R3])
    W = WildTensorHead('W', unordered_indices=True)
    U = WildTensorHead('U')
    assert a.matches(q) == {a: q}
    assert a.matches(-q) == {a: -q}
    assert g.matches(-q) == None
    assert g.matches(q) == {g: q}
    assert eps(p, -a, a).matches(eps(p, q, r)) == None
    assert eps(p, -b, a).matches(eps(p, q, r)) == {a: r, -b: q}
    assert eps(p, -q, r).replace(eps(a, b, c), 1) == 1
    assert W().matches(K(p) * V(q)) == {W(): K(p) * V(q)}
    assert W(a).matches(K(p)) == {a: p, W(a).head: _WildTensExpr(K(p))}
    assert W(a, p).matches(K(p) * V(q)) == {a: q, W(a, p).head: _WildTensExpr(K(p) * V(q))}
    assert W(p, q).matches(K(p) * V(q)) == {W(p, q).head: _WildTensExpr(K(p) * V(q))}
    assert W(p, q).matches(A(q, p)) == {W(p, q).head: _WildTensExpr(A(q, p))}
    assert U(p, q).matches(A(q, p)) == None
    assert (K(q) * K(p)).replace(W(q, p), 1) == 1