import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_with_measurement_keys():
    a, b = cirq.LineQubit.range(2)
    circuit = cirq.FrozenCircuit(cirq.X(a), cirq.measure(b, key='mb'), cirq.measure(a, key='ma'))
    op_base = cirq.CircuitOperation(circuit)
    op_with_keys = op_base.with_measurement_key_mapping({'ma': 'pa', 'x': 'z'})
    assert op_with_keys.base_operation() == op_base
    assert op_with_keys.measurement_key_map == {'ma': 'pa'}
    assert cirq.measurement_key_names(op_with_keys) == {'pa', 'mb'}
    assert cirq.with_measurement_key_mapping(op_base, {'ma': 'pa'}) == op_with_keys
    with pytest.raises(ValueError):
        _ = op_base.with_measurement_key_mapping({'ma': 'mb'})