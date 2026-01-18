import unittest.mock as mock
from typing import Optional
import numpy as np
import pytest
import sympy
import cirq
import cirq.circuits.circuit_operation as circuit_operation
from cirq import _compat
from cirq.circuits.circuit_operation import _full_join_string_lists
@pytest.mark.parametrize('sim', ALL_SIMULATORS)
def test_simulate_no_repetition_ids_inner(sim):
    q = cirq.LineQubit(0)
    inner = cirq.Circuit(cirq.measure(q, key='a'))
    middle = cirq.Circuit(cirq.CircuitOperation(inner.freeze(), repetitions=2, use_repetition_ids=False))
    outer_subcircuit = cirq.CircuitOperation(middle.freeze(), repetitions=2)
    circuit = cirq.Circuit(outer_subcircuit)
    result = sim.run(circuit)
    assert result.records['0:a'].shape == (1, 2, 1)
    assert result.records['1:a'].shape == (1, 2, 1)