from sympy.core.numbers import Integer
from sympy.core.symbol import Symbol
from sympy.physics.quantum.qexpr import QExpr, _qsympify_sequence
from sympy.physics.quantum.hilbert import HilbertSpace
from sympy.core.containers import Tuple
def test_qexpr_new():
    q = QExpr(0)
    assert q.label == (0,)
    assert q.hilbert_space == HilbertSpace()
    assert q.is_commutative is False
    q = QExpr(0, 1)
    assert q.label == (Integer(0), Integer(1))
    q = QExpr._new_rawargs(HilbertSpace(), Integer(0), Integer(1))
    assert q.label == (Integer(0), Integer(1))
    assert q.hilbert_space == HilbertSpace()