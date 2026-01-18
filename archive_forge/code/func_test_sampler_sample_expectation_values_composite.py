from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_sample_expectation_values_composite():
    q = cirq.LineQubit.range(3)
    t = [sympy.Symbol(f't{x}') for x in range(3)]
    sampler = cirq.Simulator(seed=1)
    circuit = cirq.Circuit(cirq.X(q[0]) ** t[0], cirq.X(q[1]) ** t[1], cirq.X(q[2]) ** t[2])
    obs = [cirq.Z(q[x]) for x in range(3)]
    params = ([{'t0': t0, 't1': t1, 't2': t2} for t2 in [0, 1] for t1 in [0, 1] for t0 in [0, 1]],)
    results = sampler.sample_expectation_values(circuit, obs, num_samples=5, params=params)
    assert len(results) == 8
    assert np.allclose(results, [[+1, +1, +1], [-1, +1, +1], [+1, -1, +1], [-1, -1, +1], [+1, +1, -1], [-1, +1, -1], [+1, -1, -1], [-1, -1, -1]])