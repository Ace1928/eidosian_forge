import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_parameterized_repeat_side_effects():
    q = cirq.LineQubit(0)
    op = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q).with_classical_controls('c'), cirq.measure(q, key='m')), repetitions=sympy.Symbol('a'))
    assert cirq.control_keys(op) == {cirq.MeasurementKey('c')}
    assert cirq.parameter_names(op.with_params({'a': 1})) == {'a'}
    with pytest.raises(ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'):
        cirq.measurement_key_objs(op)
    with pytest.raises(ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'):
        cirq.measurement_key_names(op)
    with pytest.raises(ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'):
        op.mapped_circuit()
    with pytest.raises(ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'):
        cirq.decompose(op)
    with pytest.raises(ValueError, match='repetition ids with parameterized repetitions'):
        op.with_repetition_ids(['x', 'y'])
    with pytest.raises(ValueError, match='repetition ids with parameterized repetitions'):
        op.repeat(repetition_ids=['x', 'y'])
    with pytest.raises(ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'):
        cirq.with_measurement_key_mapping(op, {'m': 'm2'})
    op = cirq.resolve_parameters(op, {'a': 2})
    assert set(map(str, cirq.measurement_key_objs(op))) == {'0:m', '1:m'}
    assert op.mapped_circuit() == cirq.Circuit(cirq.X(q).with_classical_controls('c'), cirq.measure(q, key=cirq.MeasurementKey.parse_serialized('0:m')), cirq.X(q).with_classical_controls('c'), cirq.measure(q, key=cirq.MeasurementKey.parse_serialized('1:m')))
    assert cirq.decompose(op) == cirq.decompose(cirq.Circuit(cirq.X(q).with_classical_controls('c'), cirq.measure(q, key=cirq.MeasurementKey.parse_serialized('0:m')), cirq.X(q).with_classical_controls('c'), cirq.measure(q, key=cirq.MeasurementKey.parse_serialized('1:m'))))