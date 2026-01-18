from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.physics.quantum.qexpr import QExpr, _qsympify_sequence
from sympy.physics.quantum.hilbert import HilbertSpace
from sympy.core.containers import Tuple
def test_qsympify():
    assert _qsympify_sequence([[1, 2], [1, 3]]) == (Tuple(1, 2), Tuple(1, 3))
    assert _qsympify_sequence(([1, 2, [3, 4, [2]], 1], 3)) == (Tuple(1, 2, Tuple(3, 4, Tuple(2)), 1), 3)
    assert _qsympify_sequence((1,)) == (1,)