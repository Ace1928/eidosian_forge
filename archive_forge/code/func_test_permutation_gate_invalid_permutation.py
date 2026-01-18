import pytest
import cirq
import numpy as np
from cirq.ops import QubitPermutationGate
def test_permutation_gate_invalid_permutation():
    with pytest.raises(ValueError, match='Invalid permutation'):
        QubitPermutationGate([1, 1])
    with pytest.raises(ValueError, match='Invalid permutation'):
        QubitPermutationGate([])