import random
from sympy.core.numbers import (Integer, Rational)
from sympy.core.singleton import S
from sympy.core.symbol import symbols
from sympy.functions.elementary.miscellaneous import sqrt
from sympy.matrices.dense import Matrix
from sympy.physics.quantum.qubit import (measure_all, measure_partial,
from sympy.physics.quantum.gate import (HadamardGate, CNOT, XGate, YGate,
from sympy.physics.quantum.qapply import qapply
from sympy.physics.quantum.represent import represent
from sympy.physics.quantum.shor import Qubit
from sympy.testing.pytest import raises
from sympy.physics.quantum.density import Density
from sympy.physics.quantum.trace import Tr
def test_superposition_of_states():
    state = 1 / sqrt(2) * Qubit('01') + 1 / sqrt(2) * Qubit('10')
    state_gate = CNOT(0, 1) * HadamardGate(0) * state
    state_expanded = Qubit('01') / 2 + Qubit('00') / 2 - Qubit('11') / 2 + Qubit('10') / 2
    assert qapply(state_gate).expand() == state_expanded
    assert matrix_to_qubit(represent(state_gate, nqubits=2)) == state_expanded