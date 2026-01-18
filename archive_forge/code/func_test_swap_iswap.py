import dataclasses
import pytest
import numpy as np
import sympy
import cirq
from cirq.transformers.eject_z import _is_swaplike
@pytest.mark.parametrize('exponent', (1, -1))
def test_swap_iswap(exponent):
    a, b = cirq.LineQubit.range(2)
    original = cirq.Circuit([cirq.rz(0.123).on(a), cirq.ISWAP(a, b) ** exponent])
    optimized = original.copy()
    optimized = cirq.eject_z(optimized)
    optimized = cirq.drop_empty_moments(optimized)
    assert optimized[0].operations == (cirq.ISWAP(a, b) ** exponent,)
    assert optimized[1].operations == (cirq.Z(b) ** (0.123 / np.pi),)
    cirq.testing.assert_allclose_up_to_global_phase(cirq.unitary(original), cirq.unitary(optimized), atol=1e-08)