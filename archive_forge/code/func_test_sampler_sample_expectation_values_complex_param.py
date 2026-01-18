from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_sample_expectation_values_complex_param():
    a = cirq.LineQubit(0)
    t = sympy.Symbol('t')
    sampler = cirq.Simulator(seed=1)
    circuit = cirq.Circuit(cirq.global_phase_operation(t))
    obs = cirq.Z(a)
    results = sampler.sample_expectation_values(circuit, [obs], num_samples=5, params=cirq.Points('t', [1, 1j, (1 + 1j) / np.sqrt(2)]))
    assert np.allclose(results, [[1], [1], [1]])