import numpy as np
import pytest
import cirq
from cirq.devices.noise_utils import (
def test_op_identifier_subtypes():
    gate_id = OpIdentifier(cirq.Gate)
    xpow_id = OpIdentifier(cirq.XPowGate)
    x_on_q0_id = OpIdentifier(cirq.XPowGate, cirq.LineQubit(0))
    assert xpow_id.is_proper_subtype_of(gate_id)
    assert x_on_q0_id.is_proper_subtype_of(xpow_id)
    assert x_on_q0_id.is_proper_subtype_of(gate_id)
    assert not xpow_id.is_proper_subtype_of(xpow_id)