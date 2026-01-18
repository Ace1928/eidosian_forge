from typing import Callable
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing as ct
from cirq import Circuit
from cirq.circuits.qasm_output import QasmUGate
from cirq.contrib.qasm_import import QasmException
from cirq.contrib.qasm_import._parser import QasmParser
def test_u3_gate():
    qasm = '\n     OPENQASM 2.0;\n     include "qelib1.inc";\n     qreg q[2];\n     u3(pi, 2.3, 3) q[0];\n     u3(+3.14, -pi, (8)) q;\n'
    parser = QasmParser()
    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')
    expected_circuit = Circuit()
    expected_circuit.append(cirq.Moment([QasmUGate(1.0, 2.3 / np.pi, 3 / np.pi)(q0), QasmUGate(3.14 / np.pi, -1.0, 8 / np.pi)(q1)]))
    expected_circuit.append(cirq.Moment([QasmUGate(3.14 / np.pi, -1.0, 8 / np.pi)(q0)]))
    parsed_qasm = parser.parse(qasm)
    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}