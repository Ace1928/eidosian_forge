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
def test_tensor_replacement():
    L = TensorIndexType('L')
    L2 = TensorIndexType('L2', dim=2)
    i, j, k, l = tensor_indices('i j k l', L)
    A, B, C, D = tensor_heads('A B C D', [L])
    H = TensorHead('H', [L, L])
    K = TensorHead('K', [L] * 4)
    expr = H(i, j)
    repl = {H(i, -j): [[1, 2], [3, 4]], L: diag(1, -1)}
    assert expr._extract_data(repl) == ([i, j], Array([[1, -2], [3, -4]]))
    assert expr.replace_with_arrays(repl) == Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl, [i, -j]) == Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [Symbol('i'), -Symbol('j')]) == Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [-i, j]) == Array([[1, -2], [-3, 4]])
    assert expr.replace_with_arrays(repl, [-i, -j]) == Array([[1, 2], [-3, -4]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[1, 3], [-2, -4]])
    assert expr.replace_with_arrays(repl, [j, -i]) == Array([[1, -3], [-2, 4]])
    assert expr.replace_with_arrays(repl, [-j, i]) == Array([[1, 3], [2, 4]])
    assert expr.replace_with_arrays(repl, [-j, -i]) == Array([[1, -3], [2, -4]])
    assert expr.replace_with_arrays(repl) == Array([[1, -2], [3, -4]])
    expr = H(i, j)
    repl = {H(i, j): [[1, 2], [3, 4]], L: diag(1, -1)}
    assert expr._extract_data(repl) == ([i, j], Array([[1, 2], [3, 4]]))
    assert expr.replace_with_arrays(repl) == Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[1, 2], [3, 4]])
    assert expr.replace_with_arrays(repl, [i, -j]) == Array([[1, -2], [3, -4]])
    assert expr.replace_with_arrays(repl, [-i, j]) == Array([[1, 2], [-3, -4]])
    assert expr.replace_with_arrays(repl, [-i, -j]) == Array([[1, -2], [-3, 4]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[1, 3], [2, 4]])
    assert expr.replace_with_arrays(repl, [j, -i]) == Array([[1, -3], [2, -4]])
    assert expr.replace_with_arrays(repl, [-j, i]) == Array([[1, 3], [-2, -4]])
    assert expr.replace_with_arrays(repl, [-j, -i]) == Array([[1, -3], [-2, 4]])
    expr = H(i, k)
    repl = {H(i, j): [[1, 2], [3, 4]], L: diag(1, -1)}
    assert expr._extract_data(repl) == ([i, k], Array([[1, 2], [3, 4]]))
    expr = A(i) * A(-i)
    repl = {A(i): [1, 2], L: diag(1, -1)}
    assert expr._extract_data(repl) == ([], -3)
    assert expr.replace_with_arrays(repl, []) == -3
    expr = K(i, j, -j, k) * A(-i) * A(-k)
    repl = {A(i): [1, 2], K(i, j, k, l): Array([1] * 2 ** 4).reshape(2, 2, 2, 2), L: diag(1, -1)}
    assert expr._extract_data(repl)
    expr = H(j, k)
    repl = {H(i, j): [[1, 2], [3, 4]], L: diag(1, -1)}
    raises(ValueError, lambda: expr._extract_data(repl))
    expr = A(i)
    repl = {B(i): [1, 2]}
    raises(ValueError, lambda: expr._extract_data(repl))
    expr = A(i)
    repl = {A(i): [[1, 2], [3, 4]]}
    raises(ValueError, lambda: expr._extract_data(repl))
    expr = A(k) * H(i, j) + B(k) * H(i, j)
    repl = {A(k): [1], B(k): [1], H(i, j): [[1, 2], [3, 4]], L: diag(1, 1)}
    assert expr._extract_data(repl) == ([k, i, j], Array([[[2, 4], [6, 8]]]))
    assert expr.replace_with_arrays(repl, [k, i, j]) == Array([[[2, 4], [6, 8]]])
    assert expr.replace_with_arrays(repl, [k, j, i]) == Array([[[2, 6], [4, 8]]])
    expr = A(k) * A(-k) + 100
    repl = {A(k): [2, 3], L: diag(1, 1)}
    assert expr.replace_with_arrays(repl, []) == 113
    expr = H(i, j) + H(j, i)
    repl = {H(i, j): [[1, 2], [3, 4]]}
    assert expr._extract_data(repl) == ([i, j], Array([[2, 5], [5, 8]]))
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[2, 5], [5, 8]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[2, 5], [5, 8]])
    expr = H(i, j) - H(j, i)
    repl = {H(i, j): [[1, 2], [3, 4]]}
    assert expr.replace_with_arrays(repl, [i, j]) == Array([[0, -1], [1, 0]])
    assert expr.replace_with_arrays(repl, [j, i]) == Array([[0, 1], [-1, 0]])
    expr = K(i, j, k, -k)
    repl = {K(i, j, k, -k): [[1, 2], [3, 4]]}
    assert expr._extract_data(repl) == ([i, j], Array([[1, 2], [3, 4]]))
    expr = H(i, -i)
    repl = {H(i, -i): 42}
    assert expr._extract_data(repl) == ([], 42)
    expr = H(i, -i)
    repl = {H(-i, -j): Array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]), L: Array([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]])}
    assert expr._extract_data(repl) == ([], 4)
    expr = A(i) * A(j)
    repl = {A(i): [1, 2]}
    raises(ValueError, lambda: expr.replace_with_arrays(repl, [j]))
    expr = A(i)
    repl = {A(i): [[1, 2]]}
    raises(ValueError, lambda: expr.replace_with_arrays(repl, [i]))
    u1, u2, u3 = tensor_indices('u1:4', L2)
    U = TensorHead('U', [L2])
    expr = U(u1) * U(-u2)
    repl = {U(u1): [[1]]}
    raises(ValueError, lambda: expr.replace_with_arrays(repl, [u1, -u2]))