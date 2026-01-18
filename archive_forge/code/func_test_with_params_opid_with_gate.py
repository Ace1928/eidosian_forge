from typing import Dict, List, Tuple
from cirq.ops.fsim_gate import PhasedFSimGate
import numpy as np
import pytest
import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
from cirq_google.devices.google_noise_properties import (
def test_with_params_opid_with_gate():
    q0, q1 = cirq.LineQubit.range(2)
    props = sample_noise_properties([q0, q1], [(q0, q1), (q1, q0)])
    expected_vals = {'gate_pauli_errors': 0.096, 'fsim_errors': cirq.PhasedFSimGate(0.0971, 0.0972, 0.0973, 0.0974, 0.0975)}
    props_v2 = props.with_params(gate_pauli_errors={cirq.PhasedXZGate: expected_vals['gate_pauli_errors']}, fsim_errors={cirq.CZPowGate: expected_vals['fsim_errors']})
    assert props_v2 != props
    gpe_op_id_0 = cirq.OpIdentifier(cirq.PhasedXZGate, q0)
    gpe_op_id_1 = cirq.OpIdentifier(cirq.PhasedXZGate, q1)
    assert props_v2.gate_pauli_errors[gpe_op_id_0] == expected_vals['gate_pauli_errors']
    assert props_v2.gate_pauli_errors[gpe_op_id_1] == expected_vals['gate_pauli_errors']
    fsim_op_id_0 = cirq.OpIdentifier(cirq.CZPowGate, q0, q1)
    fsim_op_id_1 = cirq.OpIdentifier(cirq.CZPowGate, q1, q0)
    assert props_v2.fsim_errors[fsim_op_id_0] == expected_vals['fsim_errors']
    assert props_v2.fsim_errors[fsim_op_id_1] == expected_vals['fsim_errors']