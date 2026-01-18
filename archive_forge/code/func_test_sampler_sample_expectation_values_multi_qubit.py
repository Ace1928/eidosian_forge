from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_sample_expectation_values_multi_qubit():
    q = cirq.LineQubit.range(3)
    sampler = cirq.Simulator(seed=1)
    circuit = cirq.Circuit(cirq.X(q[0]), cirq.X(q[1]), cirq.X(q[2]))
    obs = cirq.Z(q[0]) + cirq.Z(q[1]) + cirq.Z(q[2])
    results = sampler.sample_expectation_values(circuit, [obs], num_samples=5)
    assert np.allclose(results, [[-3]])