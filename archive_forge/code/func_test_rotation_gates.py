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
@pytest.mark.parametrize('qasm_gate,cirq_gate', rotation_gates)
def test_rotation_gates(qasm_gate: str, cirq_gate: Callable[[float], cirq.Gate]):
    qasm = f'OPENQASM 2.0;\n     include "qelib1.inc";\n     qreg q[2];\n     {qasm_gate}(pi/2) q[0];\n     {qasm_gate}(pi) q;\n    '
    parser = QasmParser()
    q0 = cirq.NamedQubit('q_0')
    q1 = cirq.NamedQubit('q_1')
    expected_circuit = Circuit()
    expected_circuit.append(cirq.Moment([cirq_gate(np.pi / 2).on(q0), cirq_gate(np.pi).on(q1)]))
    expected_circuit.append(cirq.Moment([cirq_gate(np.pi).on(q0)]))
    parsed_qasm = parser.parse(qasm)
    assert parsed_qasm.supportedFormat
    assert parsed_qasm.qelib1Include
    ct.assert_same_circuits(parsed_qasm.circuit, expected_circuit)
    assert parsed_qasm.qregs == {'q': 2}