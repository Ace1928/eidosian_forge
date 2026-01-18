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
def test_run_qudits_repetitions_measurement_not_terminal(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQid.for_qid_shape((2, 3))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    with mock.patch.object(simulator, '_core_iterator', wraps=simulator._core_iterator) as mock_sim:
        for b0 in [0, 1]:
            for b1 in [0, 1, 2]:
                circuit = cirq.Circuit((cirq.X ** b0)(q0), cirq.XPowGate(dimension=3)(q1) ** b1, cirq.measure(q0), cirq.measure(q1), cirq.H(q0), cirq.XPowGate(dimension=3)(q1) ** (-b1))
                result = simulator.run(circuit, repetitions=3)
                np.testing.assert_equal(result.measurements, {'q(0) (d=2)': [[b0]] * 3, 'q(1) (d=3)': [[b1]] * 3})
                assert result.repetitions == 3
        assert mock_sim.call_count == 24