import collections
import numpy as np
import pytest
import sympy
import cirq
import cirq.testing
def test_ps_initial_state_dmat():
    q0, q1 = cirq.LineQubit.range(2)
    s00 = cirq.KET_ZERO(q0) * cirq.KET_ZERO(q1)
    sp0 = cirq.KET_PLUS(q0) * cirq.KET_ZERO(q1)
    np.testing.assert_allclose(cirq.final_density_matrix(cirq.Circuit(cirq.I.on_each(q0, q1))), cirq.final_density_matrix(cirq.Circuit(cirq.I.on_each(q0, q1)), initial_state=s00))
    np.testing.assert_allclose(cirq.final_density_matrix(cirq.Circuit(cirq.H(q0), cirq.I(q1))), cirq.final_density_matrix(cirq.Circuit(cirq.I.on_each(q0, q1)), initial_state=sp0))