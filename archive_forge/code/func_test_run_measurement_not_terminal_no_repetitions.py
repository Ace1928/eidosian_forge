import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('split', [True, False])
def test_run_measurement_not_terminal_no_repetitions(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    with mock.patch.object(simulator, '_core_iterator', wraps=simulator._core_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                circuit = cirq.Circuit((cirq.X ** b0)(q0), (cirq.X ** b1)(q1), cirq.measure(q0), cirq.measure(q1), cirq.H(q0), cirq.H(q1))
                result = simulator.run(circuit, repetitions=0)
                np.testing.assert_equal(result.measurements, {'q(0)': np.empty([0, 1]), 'q(1)': np.empty([0, 1])})
                assert result.repetitions == 0
        assert mock_sim.call_count == 0