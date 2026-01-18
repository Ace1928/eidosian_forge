import numpy as np
import pytest
from pyquil import Program
from pyquil.simulation.tools import program_unitary
from cirq import Circuit, LineQubit
from cirq_rigetti.quil_input import (
from cirq.ops import (
def test_quil_with_defgate():
    q0 = LineQubit(0)
    cirq_circuit = Circuit([X(q0), Z(q0)])
    quil_circuit = circuit_from_quil(QUIL_PROGRAM_WITH_DEFGATE)
    assert np.isclose(quil_circuit.unitary(), cirq_circuit.unitary()).all()