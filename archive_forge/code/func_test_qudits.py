import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_qudits():
    q0, q1 = cirq.LineQid.range(2, dimension=3)
    circuit = cirq.Circuit(cirq.measure(q0, key='a'), cirq.XPowGate(dimension=3).on(q1).with_classical_controls('a'), cirq.measure(q1, key='b'))
    assert_equivalent_to_deferred(circuit)
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(deferred, cirq.Circuit(_mod_add(q0, q_ma), cirq.XPowGate(dimension=3).on(q1).controlled_by(q_ma, control_values=[[1, 2]]), cirq.measure(q_ma, key='a'), cirq.measure(q1, key='b')))