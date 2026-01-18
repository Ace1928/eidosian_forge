from typing import List
import pytest
import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
def test_two_qubit_compilation_decompose_operation_not_implemented():
    gateset = ExampleCXTargetGateset()
    q = cirq.LineQubit.range(3)
    assert gateset.decompose_to_target_gateset(cirq.measure(q[0]), 1) is NotImplemented
    assert gateset.decompose_to_target_gateset(cirq.measure(*q[:2]), 1) is NotImplemented
    assert gateset.decompose_to_target_gateset(cirq.X(q[0]).with_classical_controls('m'), 1) is NotImplemented
    assert gateset.decompose_to_target_gateset(cirq.CCZ(*q), 1) is NotImplemented