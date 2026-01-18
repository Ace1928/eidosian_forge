from sympy.core import symbols, S, Pow, Function
from sympy.functions import exp
from sympy.testing.pytest import raises
from sympy.tensor.indexed import Idx, IndexedBase
from sympy.tensor.index_methods import IndexConformanceException
from sympy.tensor.index_methods import (get_contraction_structure, get_indices)
def test_trivial_indices():
    x, y = symbols('x y')
    assert get_indices(x) == (set(), {})
    assert get_indices(x * y) == (set(), {})
    assert get_indices(x + y) == (set(), {})
    assert get_indices(x ** y) == (set(), {})