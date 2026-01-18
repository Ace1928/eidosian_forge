import functools
import itertools
from typing import Tuple, Type
import numpy as np
import pytest
import cirq
from cirq.protocols.act_on_protocol_test import ExampleSimulationState
from cirq.testing import EqualsTester, assert_allclose_up_to_global_phase
@pytest.mark.parametrize('trans_x,trans_z', _all_rotation_pairs())
def test_to_phased_xz_gate(trans_x, trans_z):
    gate = cirq.SingleQubitCliffordGate.from_xz_map(trans_x, trans_z)
    actual_phased_xz_gate = gate.to_phased_xz_gate()._canonical()
    expect_phased_xz_gates = cirq.PhasedXZGate.from_matrix(cirq.unitary(gate))
    assert np.isclose(actual_phased_xz_gate.x_exponent, expect_phased_xz_gates.x_exponent)
    assert np.isclose(actual_phased_xz_gate.z_exponent, expect_phased_xz_gates.z_exponent)
    assert np.isclose(actual_phased_xz_gate.axis_phase_exponent, expect_phased_xz_gates.axis_phase_exponent)