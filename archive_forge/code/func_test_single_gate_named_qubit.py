import os
import numpy as np
import pytest
import cirq
from cirq.ops.pauli_interaction_gate import PauliInteractionGate
import cirq_rigetti
from cirq_rigetti.quil_output import QuilOutput
def test_single_gate_named_qubit():
    q = cirq.NamedQubit('qTest')
    output = cirq_rigetti.quil_output.QuilOutput((cirq.X(q),), (q,))
    assert str(output) == '# Created using Cirq.\n\nX 0\n'