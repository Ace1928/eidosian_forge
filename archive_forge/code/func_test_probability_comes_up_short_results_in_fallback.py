from typing import cast, Type
from unittest import mock
import numpy as np
import pytest
import cirq
def test_probability_comes_up_short_results_in_fallback():

    class Short(cirq.Gate):

        def num_qubits(self) -> int:
            return 1

        def _kraus_(self):
            return [cirq.unitary(cirq.X) * np.sqrt(0.999), np.eye(2) * 0]
    mock_prng = mock.Mock()
    mock_prng.random.return_value = 0.9999
    args = cirq.StateVectorSimulationState(available_buffer=np.empty(2, dtype=np.complex64), qubits=cirq.LineQubit.range(1), prng=mock_prng, initial_state=np.array([1, 0], dtype=np.complex64), dtype=np.complex64)
    cirq.act_on(Short(), args, cirq.LineQubit.range(1))
    np.testing.assert_allclose(args.target_tensor, np.array([0, 1]))