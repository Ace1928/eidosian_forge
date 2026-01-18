import itertools
import random
from typing import Type
from unittest import mock
import numpy as np
import pytest
import sympy
import cirq
def test_state_vector_copy():
    sim = cirq.Simulator(split_untangled_states=False)

    class InplaceGate(cirq.testing.SingleQubitGate):
        """A gate that modifies the target tensor in place, multiply by -1."""

        def _apply_unitary_(self, args):
            args.target_tensor *= -1.0
            return args.target_tensor
    q = cirq.LineQubit(0)
    circuit = cirq.Circuit(InplaceGate()(q), InplaceGate()(q))
    vectors = []
    for step in sim.simulate_moment_steps(circuit):
        vectors.append(step.state_vector(copy=True))
    for x, y in itertools.combinations(vectors, 2):
        assert not np.shares_memory(x, y)
    vectors = []
    copy_of_vectors = []
    for step in sim.simulate_moment_steps(circuit):
        state_vector = step.state_vector()
        vectors.append(state_vector)
        copy_of_vectors.append(state_vector.copy())
    assert any((not np.array_equal(x, y) for x, y in zip(vectors, copy_of_vectors)))