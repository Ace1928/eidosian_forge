from typing import cast
import itertools
import cmath
import pytest
import numpy as np
from cirq.ops import DensePauliString, T
from cirq import protocols
from cirq.transformers.analytical_decompositions import unitary_to_pauli_string
@pytest.mark.parametrize('phase', [cmath.exp(i * 2 * cmath.pi / 5 * 1j) for i in range(5)])
@pytest.mark.parametrize('pauli_string', [''.join(p) for p in itertools.product(['', 'I', 'X', 'Y', 'Z'], repeat=4)])
def test_unitary_to_pauli_string(pauli_string: str, phase: complex):
    want = DensePauliString(pauli_string, coefficient=phase)
    got = unitary_to_pauli_string(protocols.unitary(want))
    assert got is not None
    assert np.all(want.pauli_mask == got.pauli_mask)
    assert np.isclose(cast(np.complex128, want.coefficient), cast(np.complex128, got.coefficient))