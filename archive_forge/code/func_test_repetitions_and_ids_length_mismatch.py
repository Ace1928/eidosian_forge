import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
def test_repetitions_and_ids_length_mismatch():
    a, b, c = cirq.LineQubit.range(3)
    circuit = cirq.FrozenCircuit(cirq.X(a), cirq.Y(b), cirq.H(c), cirq.CX(a, b) ** sympy.Symbol('exp'), cirq.measure(a, b, c, key='m'))
    with pytest.raises(ValueError, match='Expected repetition_ids to be a list of length 2'):
        _ = cirq.CircuitOperation(circuit, repetitions=2, repetition_ids=['a', 'b', 'c'])