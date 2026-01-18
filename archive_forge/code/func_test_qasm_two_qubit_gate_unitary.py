import re
import os
import numpy as np
import pytest
import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq
def test_qasm_two_qubit_gate_unitary():
    u = cirq.testing.random_unitary(4)
    g = QasmTwoQubitGate.from_matrix(u)
    np.testing.assert_allclose(cirq.unitary(g), u)