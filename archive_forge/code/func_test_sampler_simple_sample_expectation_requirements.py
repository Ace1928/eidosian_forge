from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_simple_sample_expectation_requirements():
    a = cirq.LineQubit(0)
    sampler = cirq.Simulator(seed=1)
    circuit = cirq.Circuit(cirq.H(a))
    obs = cirq.X(a)
    with pytest.raises(ValueError, match='at least one sample'):
        _ = sampler.sample_expectation_values(circuit, [obs], num_samples=0)
    with pytest.raises(ValueError, match='At least one observable'):
        _ = sampler.sample_expectation_values(circuit, [], num_samples=1)
    circuit.append(cirq.measure(a, key='out'))
    with pytest.raises(ValueError, match='permit_terminal_measurements'):
        _ = sampler.sample_expectation_values(circuit, [obs], num_samples=1)