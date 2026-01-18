import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
def test_qasm_multiple_conditions():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.measure(q0, key='b'), cirq.X(q1).with_classical_controls(sympy.Eq(sympy.Symbol('a'), 0), sympy.Eq(sympy.Symbol('b'), 0)))
    with pytest.raises(ValueError, match='QASM does not support multiple conditions'):
        _ = cirq.qasm(circuit)