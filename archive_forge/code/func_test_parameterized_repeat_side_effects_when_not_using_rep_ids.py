import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_parameterized_repeat_side_effects_when_not_using_rep_ids():
    q = cirq.LineQubit(0)
    op = cirq.CircuitOperation(cirq.FrozenCircuit(cirq.X(q).with_classical_controls('c'), cirq.measure(q, key='m')), repetitions=sympy.Symbol('a'), use_repetition_ids=False)
    assert cirq.control_keys(op) == {cirq.MeasurementKey('c')}
    assert cirq.parameter_names(op.with_params({'a': 1})) == {'a'}
    assert set(map(str, cirq.measurement_key_objs(op))) == {'m'}
    assert cirq.measurement_key_names(op) == {'m'}
    assert cirq.measurement_key_names(cirq.with_measurement_key_mapping(op, {'m': 'm2'})) == {'m2'}
    with pytest.raises(ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'):
        op.mapped_circuit()
    with pytest.raises(ValueError, match='Cannot unroll circuit due to nondeterministic repetitions'):
        cirq.decompose(op)
    with pytest.raises(ValueError, match='repetition ids with parameterized repetitions'):
        op.with_repetition_ids(['x', 'y'])
    with pytest.raises(ValueError, match='repetition ids with parameterized repetitions'):
        op.repeat(repetition_ids=['x', 'y'])