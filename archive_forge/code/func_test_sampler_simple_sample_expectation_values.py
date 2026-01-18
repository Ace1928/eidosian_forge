from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_simple_sample_expectation_values():
    a = cirq.LineQubit(0)
    sampler = cirq.Simulator()
    circuit = cirq.Circuit(cirq.H(a))
    obs = cirq.X(a)
    results = sampler.sample_expectation_values(circuit, [obs], num_samples=1000)
    assert np.allclose(results, [[1]])