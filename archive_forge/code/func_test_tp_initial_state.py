import numpy as np
import pytest
import cirq
def test_tp_initial_state():
    q0, q1 = cirq.LineQubit.range(2)
    psi1 = cirq.final_state_vector(cirq.Circuit([cirq.I.on_each(q0, q1), cirq.X(q1)]))
    s01 = cirq.KET_ZERO(q0) * cirq.KET_ONE(q1)
    psi2 = cirq.final_state_vector(cirq.Circuit([cirq.I.on_each(q0, q1)]), initial_state=s01)
    np.testing.assert_allclose(psi1, psi2)