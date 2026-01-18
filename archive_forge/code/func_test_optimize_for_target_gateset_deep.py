import cirq
from cirq.protocols.decompose_protocol import DecomposeResult
from cirq.transformers.optimize_for_target_gateset import _decompose_operations_to_target_gateset
import pytest
def test_optimize_for_target_gateset_deep():
    q0, q1 = cirq.LineQubit.range(2)
    c_nested = cirq.FrozenCircuit(cirq.CX(q0, q1))
    c_orig = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.H(q0), cirq.CircuitOperation(c_nested).repeat(3))).repeat(5))
    c_expected = cirq.Circuit(cirq.CircuitOperation(cirq.FrozenCircuit(cirq.single_qubit_matrix_to_phxz(cirq.unitary(cirq.H(q0))).on(q0), cirq.CircuitOperation(cirq.FrozenCircuit(cirq.MatrixGate(c_nested.unitary(qubit_order=[q0, q1]), name='M').on(q0, q1))).repeat(3))).repeat(5))
    gateset = MatrixGateTargetGateset()
    context = cirq.TransformerContext(deep=True)
    c_new = cirq.optimize_for_target_gateset(c_orig, gateset=gateset, context=context)
    cirq.testing.assert_circuits_with_terminal_measurements_are_equivalent(c_new, c_expected)
    cirq.testing.assert_has_diagram(c_orig, '\n      [           [ 0: ───@─── ]             ]\n      [ 0: ───H───[       │    ]──────────── ]\n0: ───[           [ 1: ───X─── ](loops=3)    ]────────────\n      [           │                          ]\n      [ 1: ───────#2──────────────────────── ](loops=5)\n      │\n1: ───#2──────────────────────────────────────────────────\n')
    cirq.testing.assert_has_diagram(c_new, '\n      [                                 [ 0: ───M[1]─── ]             ]\n      [ 0: ───PhXZ(a=-0.5,x=0.5,z=-1)───[       │       ]──────────── ]\n0: ───[                                 [ 1: ───M[2]─── ](loops=3)    ]────────────\n      [                                 │                             ]\n      [ 1: ─────────────────────────────#2─────────────────────────── ](loops=5)\n      │\n1: ───#2───────────────────────────────────────────────────────────────────────────\n')