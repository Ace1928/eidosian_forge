import numpy as np
import pytest
import sympy
from scipy import linalg
import cirq
def test_swap_has_stabilizer_effect():
    assert cirq.has_stabilizer_effect(cirq.SWAP)
    assert cirq.has_stabilizer_effect(cirq.SWAP ** 2)
    assert not cirq.has_stabilizer_effect(cirq.SWAP ** 0.5)
    assert not cirq.has_stabilizer_effect(cirq.SWAP ** sympy.Symbol('foo'))