import pytest
import cirq
import numpy as np
from cirq.ops import QubitPermutationGate
def test_permutation_gate_equality():
    eq = cirq.testing.EqualsTester()
    eq.make_equality_group(lambda: QubitPermutationGate([0, 1]), lambda: QubitPermutationGate((0, 1)))
    eq.add_equality_group(QubitPermutationGate([1, 0]), QubitPermutationGate((1, 0)))