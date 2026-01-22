from typing import Sequence
import pytest
import numpy as np
import cirq
class NoisySingleQubitReadoutSampler(cirq.Sampler):

    def __init__(self, p0: float, p1: float, seed: 'cirq.RANDOM_STATE_OR_SEED_LIKE'=None):
        """Sampler that flips some bits upon readout.

        Args:
            p0: Probability of flipping a 0 to a 1.
            p1: Probability of flipping a 1 to a 0.
            seed: A seed for the pseudorandom number generator.
        """
        self.p0 = p0
        self.p1 = p1
        self.prng = cirq.value.parse_random_state(seed)
        self.simulator = cirq.Simulator(seed=self.prng, split_untangled_states=False)

    def run_sweep(self, program: 'cirq.AbstractCircuit', params: cirq.Sweepable, repetitions: int=1) -> Sequence[cirq.Result]:
        results = self.simulator.run_sweep(program, params, repetitions)
        for result in results:
            for bits in result.measurements.values():
                rand_num = self.prng.uniform(size=bits.shape)
                should_flip = np.logical_or(np.logical_and(bits == 0, rand_num < self.p0), np.logical_and(bits == 1, rand_num < self.p1))
                bits[should_flip] ^= 1
        return results