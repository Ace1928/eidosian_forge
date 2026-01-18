from sympy.core.numbers import (I, pi)
from sympy.core.symbol import Symbol
from sympy.functions.elementary.exponential import exp
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.qft import QFT, IQFT, RkGate
from sympy.physics.quantum.gate import (ZGate, SwapGate, HadamardGate, CGate,
from sympy.physics.quantum.qubit import Qubit
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
def test_qft_represent():
    c = QFT(0, 3)
    a = represent(c, nqubits=3)
    b = represent(c.decompose(), nqubits=3)
    assert a.evalf(n=10) == b.evalf(n=10)