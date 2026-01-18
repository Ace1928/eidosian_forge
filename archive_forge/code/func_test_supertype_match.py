from typing import Dict, List, Tuple
from cirq.ops.fsim_gate import PhasedFSimGate
import numpy as np
import pytest
import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
from cirq_google.devices.google_noise_properties import (
def test_supertype_match():
    q0, q1 = cirq.LineQubit.range(2)
    op_id = OpIdentifier(cirq_google.SycamoreGate, q0, q1)
    test_props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    expected_err = test_props._depolarizing_error[op_id]
    props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    props.fsim_errors = {k: cirq.PhasedFSimGate(0.5, 0.4, 0.3, 0.2, 0.1) for k in [OpIdentifier(cirq.FSimGate, q0, q1), OpIdentifier(cirq.FSimGate, q1, q0)]}
    assert props._depolarizing_error[op_id] != expected_err