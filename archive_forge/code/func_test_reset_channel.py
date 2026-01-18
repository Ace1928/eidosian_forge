import re
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
def test_reset_channel():
    r = cirq.reset(cirq.LineQubit(0))
    np.testing.assert_almost_equal(cirq.kraus(r), (np.array([[1.0, 0.0], [0.0, 0]]), np.array([[0.0, 1.0], [0.0, 0.0]])))
    cirq.testing.assert_consistent_channel(r)
    assert not cirq.has_mixture(r)
    assert cirq.num_qubits(r) == 1
    assert cirq.qid_shape(r) == (2,)
    r = cirq.reset(cirq.LineQid(0, dimension=3))
    np.testing.assert_almost_equal(cirq.kraus(r), (np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]), np.array([[0, 1, 0], [0, 0, 0], [0, 0, 0]]), np.array([[0, 0, 1], [0, 0, 0], [0, 0, 0]])))
    cirq.testing.assert_consistent_channel(r)
    assert not cirq.has_mixture(r)
    assert cirq.qid_shape(r) == (3,)