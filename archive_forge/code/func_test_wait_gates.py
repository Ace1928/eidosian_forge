from typing import Dict, List, Tuple
from cirq.ops.fsim_gate import PhasedFSimGate
import numpy as np
import pytest
import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
from cirq_google.devices.google_noise_properties import (
def test_wait_gates():
    q0 = cirq.LineQubit(0)
    props = sample_noise_properties([q0], [])
    model = NoiseModelFromGoogleNoiseProperties(props)
    op = cirq.wait(q0, nanos=100)
    circuit = cirq.Circuit(op)
    noisy_circuit = circuit.with_noise(model)
    assert len(noisy_circuit.moments) == 2
    assert noisy_circuit.moments[0].operations[0] == op.with_tags(PHYSICAL_GATE_TAG)
    assert len(noisy_circuit.moments[1].operations) == 1
    thermal_op = noisy_circuit.moments[1].operations[0]
    assert isinstance(thermal_op.gate, cirq.KrausChannel)
    thermal_choi = cirq.kraus_to_choi(cirq.kraus(thermal_op))
    assert np.allclose(thermal_choi, [[1, 0, 0, 0.9990005], [0, 0.000999500167, 0, 0], [0, 0, 0, 0], [0.9990005, 0, 0, 0.9990005]])