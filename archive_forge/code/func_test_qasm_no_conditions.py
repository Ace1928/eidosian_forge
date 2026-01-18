import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_qasm_no_conditions():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.ClassicallyControlledOperation(cirq.X(q1), []))
    qasm = cirq.qasm(circuit)
    assert qasm == f'// Generated from Cirq v{cirq.__version__}\n\nOPENQASM 2.0;\ninclude "qelib1.inc";\n\n\n// Qubits: [q(0), q(1)]\nqreg q[2];\ncreg m_a[1];\n\n\nmeasure q[0] -> m_a[0];\nx q[1];\n'