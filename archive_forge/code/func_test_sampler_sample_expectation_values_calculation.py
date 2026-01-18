from typing import Sequence
import pytest
import duet
import numpy as np
import pandas as pd
import sympy
import cirq
def test_sampler_sample_expectation_values_calculation():

    class DeterministicImbalancedStateSampler(cirq.Sampler):
        """A simple, deterministic mock sampler.
        Pretends to sample from a state vector with a 3:1 balance between the
        probabilities of the |0) and |1) state.
        """

        def run_sweep(self, program: 'cirq.AbstractCircuit', params: 'cirq.Sweepable', repetitions: int=1) -> Sequence['cirq.Result']:
            results = np.zeros((repetitions, 1), dtype=bool)
            for idx in range(repetitions // 4):
                results[idx][0] = 1
            return [cirq.ResultDict(params=pr, measurements={'z': results}) for pr in cirq.study.to_resolvers(params)]
    a = cirq.LineQubit(0)
    sampler = DeterministicImbalancedStateSampler()
    circuit = cirq.Circuit(cirq.X(a) ** (1 / 3))
    obs = cirq.Z(a)
    results = sampler.sample_expectation_values(circuit, [obs], num_samples=1000)
    assert np.allclose(results, [[0.5]])