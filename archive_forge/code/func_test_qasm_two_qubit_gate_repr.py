import re
import os
import numpy as np
import pytest
import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq
def test_qasm_two_qubit_gate_repr():
    cirq.testing.assert_equivalent_repr(QasmTwoQubitGate.from_matrix(cirq.testing.random_unitary(4)))