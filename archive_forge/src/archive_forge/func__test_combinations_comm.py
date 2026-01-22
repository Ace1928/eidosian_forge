from sympy.unify.core import Compound, Variable, CondVariable, allcombinations
from sympy.unify import core
def _test_combinations_comm():
    assert set(allcombinations((1, 2, 3), (a, b), None)) == {(((1,), (2, 3)), ('a', 'b')), (((2,), (3, 1)), ('a', 'b')), (((3,), (1, 2)), ('a', 'b')), (((1, 2), (3,)), ('a', 'b')), (((2, 3), (1,)), ('a', 'b')), (((3, 1), (2,)), ('a', 'b'))}