import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
def test_swap_gate_eq():
    assert cca.SwapPermutationGate() == cca.SwapPermutationGate(cirq.SWAP)
    assert cca.SwapPermutationGate() != cca.SwapPermutationGate(cirq.CZ)
    assert cca.SwapPermutationGate(cirq.CZ) == cca.SwapPermutationGate(cirq.CZ)