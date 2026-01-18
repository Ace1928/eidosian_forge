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
def test_quantum_fourier():
    assert QFT(0, 3).decompose() == SwapGate(0, 2) * HadamardGate(0) * CGate((0,), PhaseGate(1)) * HadamardGate(1) * CGate((0,), TGate(2)) * CGate((1,), PhaseGate(2)) * HadamardGate(2)
    assert IQFT(0, 3).decompose() == HadamardGate(2) * CGate((1,), RkGate(2, -2)) * CGate((0,), RkGate(2, -3)) * HadamardGate(1) * CGate((0,), RkGate(1, -2)) * HadamardGate(0) * SwapGate(0, 2)
    assert represent(QFT(0, 3), nqubits=3) == Matrix([[exp(2 * pi * I / 8) ** (i * j % 8) / sqrt(8) for i in range(8)] for j in range(8)])
    assert QFT(0, 4).decompose()
    assert qapply(QFT(0, 3).decompose() * Qubit(0, 0, 0)).expand() == qapply(HadamardGate(0) * HadamardGate(1) * HadamardGate(2) * Qubit(0, 0, 0)).expand()