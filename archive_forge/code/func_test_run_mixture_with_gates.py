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
def test_run_mixture_with_gates(dtype: Type[np.complexfloating], split: bool):
    q0 = cirq.LineQubit(0)
    simulator = cirq.Simulator(dtype=dtype, split_untangled_states=split, seed=23)
    circuit = cirq.Circuit(cirq.H(q0), cirq.phase_flip(0.5)(q0), cirq.H(q0), cirq.measure(q0))
    result = simulator.run(circuit, repetitions=100)
    assert sum(result.measurements['q(0)'])[0] < 80
    assert sum(result.measurements['q(0)'])[0] > 20