import pytest
import cirq
import numpy as np
from cirq.ops import QubitPermutationGate
@pytest.mark.parametrize('maps, permutation', [[{0: 0}, [0]], [{0: 0, 1: 1, 2: 2}, [0, 1, 2]], [{0: 0, 1: 4, 2: 2, 4: 1, 7: 7, 5: 5}, [2, 1, 0]]])
def test_permutation_gate_maps(maps, permutation):
    qs = cirq.LineQubit.range(len(permutation))
    permutationOp = cirq.QubitPermutationGate(permutation).on(*qs)
    circuit = cirq.Circuit(permutationOp)
    cirq.testing.assert_equivalent_computational_basis_map(maps, circuit)
    circuit = cirq.Circuit(cirq.I.on_each(*qs), cirq.decompose(permutationOp))
    cirq.testing.assert_equivalent_computational_basis_map(maps, circuit)