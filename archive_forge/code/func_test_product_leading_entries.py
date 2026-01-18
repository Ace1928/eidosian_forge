from sympy.matrices.dense import Matrix
from sympy.polys.polytools import factor
from sympy.core import symbols
from sympy.tensor.indexed import IndexedBase
from sympy.polys.multivariate_resultants import (DixonResultant,
def test_product_leading_entries():
    """Tests product of leading entries method."""
    A, B = symbols('A, B')
    m1 = Matrix([[1, 2, 3], [0, 4, 5], [0, 0, 6]])
    m2 = Matrix([[0, 0, 1], [2, 0, 3]])
    m3 = Matrix([[0, 0, 0], [1, 2, 3], [0, 0, 0]])
    m4 = Matrix([[0, 0, A], [1, 2, 3], [B, 0, 0]])
    assert dixon.product_leading_entries(m1) == 24
    assert dixon.product_leading_entries(m2) == 2
    assert dixon.product_leading_entries(m3) == 1
    assert dixon.product_leading_entries(m4) == A * B