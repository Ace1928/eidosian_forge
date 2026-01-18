import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_sympy_control_multiqubit():
    q0, q1, q2 = cirq.LineQubit.range(3)
    circuit = cirq.Circuit(cirq.measure(q0, q1, key='a'), cirq.X(q2).with_classical_controls(sympy_parser.parse_expr('a >= 2')), cirq.measure(q2, key='c'))
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma0 = _MeasurementQid('a', q0)
    q_ma1 = _MeasurementQid('a', q1)
    cirq.testing.assert_same_circuits(deferred, cirq.Circuit(cirq.CX(q0, q_ma0), cirq.CX(q1, q_ma1), cirq.ControlledOperation([q_ma0, q_ma1], cirq.X(q2), cirq.SumOfProducts([[1, 0], [1, 1]])), cirq.measure(q_ma0, q_ma1, key='a'), cirq.measure(q2, key='c')))