import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
@pytest.mark.parametrize('dtype', [np.complex64, np.complex128])
@pytest.mark.parametrize('split', [True, False])
def test_run_invert_mask_measure_not_terminal(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQubit.range(2)
    simulator = cirq.Simulator(dtype=dtype, split_untangled_states=split)
    with mock.patch.object(simulator, '_core_iterator', wraps=simulator._core_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1]:
                circuit = cirq.Circuit((cirq.X ** b0)(q0), (cirq.X ** b1)(q1), cirq.measure(q0, q1, key='m', invert_mask=(True, False)), cirq.X(q0))
                result = simulator.run(circuit, repetitions=3)
                np.testing.assert_equal(result.measurements, {'m': [[1 - b0, b1]] * 3})
                assert result.repetitions == 3
        assert mock_sim.call_count > 4