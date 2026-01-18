from typing import Dict, List, Tuple
from cirq.ops.fsim_gate import PhasedFSimGate
import numpy as np
import pytest
import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
from cirq_google.devices.google_noise_properties import (
def test_with_params_target():
    q0, q1 = cirq.LineQubit.range(2)
    props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    expected_vals = {'gate_times_ns': {cirq.ZPowGate: 91}, 't1_ns': {q0: 92}, 'tphi_ns': {q1: 93}, 'readout_errors': {q0: [0.094, 0.095]}, 'gate_pauli_errors': {cirq.OpIdentifier(cirq.PhasedXZGate, q1): 0.096}, 'fsim_errors': {cirq.OpIdentifier(cirq.CZPowGate, q0, q1): cirq.PhasedFSimGate(0.0971, 0.0972, 0.0973, 0.0974, 0.0975)}}
    props_v2 = props.with_params(gate_times_ns=expected_vals['gate_times_ns'], t1_ns=expected_vals['t1_ns'], tphi_ns=expected_vals['tphi_ns'], readout_errors=expected_vals['readout_errors'], gate_pauli_errors=expected_vals['gate_pauli_errors'], fsim_errors=expected_vals['fsim_errors'])
    assert props_v2 != props
    for field_name, expected in expected_vals.items():
        target_dict = getattr(props_v2, field_name)
        for key, val in expected.items():
            if isinstance(target_dict[key], np.ndarray):
                assert np.allclose(target_dict[key], val)
            else:
                assert target_dict[key] == val