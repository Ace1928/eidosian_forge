from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.physics.quantum.qexpr import QExpr, _qsympify_sequence
from sympy.physics.quantum.hilbert import HilbertSpace
from sympy.core.containers import Tuple
def test_qexpr_commutative_free_symbols():
    q1 = QExpr(x)
    assert q1.free_symbols.pop().is_commutative is False
    q2 = QExpr('q2')
    assert q2.free_symbols.pop().is_commutative is False