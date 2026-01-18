import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_repeat_until_error():
    q = cirq.LineQubit(0)
    with pytest.raises(ValueError, match='Cannot use repetitions with repeat_until'):
        cirq.CircuitOperation(cirq.FrozenCircuit(), use_repetition_ids=True, repeat_until=cirq.KeyCondition(cirq.MeasurementKey('a')))
    with pytest.raises(ValueError, match='Infinite loop'):
        cirq.CircuitOperation(cirq.FrozenCircuit(cirq.measure(q, key='m')), use_repetition_ids=False, repeat_until=cirq.KeyCondition(cirq.MeasurementKey('a')))