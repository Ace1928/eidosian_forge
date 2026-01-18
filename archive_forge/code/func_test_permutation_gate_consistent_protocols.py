import pytest
import cirq
import numpy as np
from cirq.ops import QubitPermutationGate
@pytest.mark.parametrize('permutation', [rs.permutation(i) for i in range(3, 7)])
def test_permutation_gate_consistent_protocols(permutation):
    gate = QubitPermutationGate(list(permutation))
    cirq.testing.assert_implements_consistent_protocols(gate)