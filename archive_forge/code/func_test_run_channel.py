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
def test_run_channel(dtype: Type[np.complexfloating], split: bool):
    q0, q1 = cirq.LineQubit.range(2)
    circuit = cirq.Circuit(cirq.X(q0), cirq.amplitude_damp(0.5)(q0), cirq.measure(q0), cirq.measure(q1))
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    result = simulator.run(circuit, repetitions=100)
    np.testing.assert_equal(result.measurements['q(1)'], [[0]] * 100)
    q0_measurements = set((x[0] for x in result.measurements['q(0)'].tolist()))
    assert q0_measurements == {0, 1}