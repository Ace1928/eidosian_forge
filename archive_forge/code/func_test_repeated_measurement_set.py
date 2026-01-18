import numpy as np
import pytest
import sympy
from sympy.parsing import sympy_parser
import cirq
@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_repeated_measurement_set(sim):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.measure(q0, key='a'), cirq.X(q0), cirq.measure(q0, key='a'), cirq.X(q1).with_classical_controls(cirq.KeyCondition(cirq.MeasurementKey('a'), index=-2)), cirq.measure(q1, key='b'), cirq.X(q1).with_classical_controls(cirq.KeyCondition(cirq.MeasurementKey('a'), index=-1)), cirq.measure(q1, key='c'))
    result = sim.run(circuit)
    assert result.records['a'][0][0][0] == 1
    assert result.records['a'][0][1][0] == 0
    assert result.records['b'][0][0][0] == 1
    assert result.records['c'][0][0][0] == 1