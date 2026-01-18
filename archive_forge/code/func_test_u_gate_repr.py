import re
import os
import numpy as np
import pytest
import cirq
from cirq.circuits.qasm_output import QasmTwoQubitGate, QasmUGate
from cirq.testing import consistent_qasm as cq
def test_u_gate_repr():
    gate = QasmUGate(0.1, 0.2, 0.3)
    assert repr(gate) == 'cirq.circuits.qasm_output.QasmUGate(theta=0.1, phi=0.2, lmda=0.3)'