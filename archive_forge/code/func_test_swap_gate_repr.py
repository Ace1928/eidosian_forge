import random
import pytest
import cirq
import cirq.testing as ct
import cirq.contrib.acquaintance as cca
@pytest.mark.parametrize('gate', [cca.SwapPermutationGate(), cca.SwapPermutationGate(cirq.SWAP), cca.SwapPermutationGate(cirq.CZ)])
def test_swap_gate_repr(gate):
    cirq.testing.assert_equivalent_repr(gate)