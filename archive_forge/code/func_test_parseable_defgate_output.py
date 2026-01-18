import os
import numpy as np
import pytest
import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate
import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput
def test_parseable_defgate_output():
    pyquil = pytest.importorskip('pyquil')
    q0, q1 = _make_qubits(2)
    operations = [cirq_rigetti.quil_output.QuilOneQubitGate(np.array([[1, 0], [0, 1]])).on(q0), cirq_rigetti.quil_output.QuilTwoQubitGate(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])).on(q0, q1)]
    output = cirq_rigetti.quil_output.QuilOutput(operations, (q0, q1))
    pyquil.Program(str(output))