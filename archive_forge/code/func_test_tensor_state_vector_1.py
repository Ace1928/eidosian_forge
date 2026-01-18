import functools
import operator
import numpy as np
import pytest
import cirq
import cirq.contrib.quimb as ccq
def test_tensor_state_vector_1():
    q = cirq.LineQubit.range(2)
    c = cirq.Circuit(cirq.YPowGate(exponent=0.25).on(q[0]))
    psi1 = cirq.final_state_vector(c, qubit_order=q, dtype=np.complex128)
    psi2 = ccq.tensor_state_vector(c, q)
    np.testing.assert_allclose(psi1, psi2, atol=1e-15)