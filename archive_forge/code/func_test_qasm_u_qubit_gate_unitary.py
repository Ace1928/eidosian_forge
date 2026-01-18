import re
import os
import numpy as np
import pytest
import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq
def test_qasm_u_qubit_gate_unitary():
    u = cirq.testing.random_unitary(2)
    g = QasmUGate.from_matrix(u)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(g), u, atol=1e-07)
    cirq.testing.assert_implements_consistent_protocols(g)