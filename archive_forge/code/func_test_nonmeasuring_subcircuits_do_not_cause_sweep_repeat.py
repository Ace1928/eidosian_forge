import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_nonmeasuring_subcircuits_do_not_cause_sweep_repeat():
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(cirq.CircuitOperation(cirq.Circuit(cirq.H(q)).freeze()), cirq.measure(q, key='x'))
    simulator = cirq.DensityMatrixSimulator()
    with mock.patch.object(simulator, '_core_iterator', wraps=simulator._core_iterator) as mock_sim:
        simulator.run(circuit, repetitions=10)
        assert mock_sim.call_count == 2