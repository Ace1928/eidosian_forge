import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
from cirq.transformers.measurement_transformers import _ConfusionChannel, _MeasurementQid, _mod_add
def test_pauli():
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.PauliMeasurementGate(cirq.DensePauliString('Y'), key='a').on(q0), cirq.X(q1).with_classical_controls('a'), cirq.measure(q1, key='b'))
    deferred = cirq.defer_measurements(circuit)
    q_ma = _MeasurementQid('a', q0)
    cirq.testing.assert_same_circuits(cirq.unroll_circuit_op(deferred), cirq.Circuit(cirq.SingleQubitCliffordGate.X_sqrt(q0), cirq.CX(q0, q_ma), cirq.SingleQubitCliffordGate.X_sqrt(q0) ** (-1), cirq.Moment(cirq.CX(q_ma, q1)), cirq.measure(q_ma, key='a'), cirq.measure(q1, key='b')))