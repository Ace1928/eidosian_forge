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
def test_simulate_moment_steps_empty_circuit(dtype: Type[np.complexfloating], split: bool):
    circuit = cirq.Circuit()
    simulator = cirq.DensityMatrixSimulator(dtype=dtype, split_untangled_states=split)
    step = None
    for step in simulator.simulate_moment_steps(circuit):
        pass
    assert np.allclose(step.density_matrix(), np.array([[1]]))
    assert not cirq.qid_shape(step)