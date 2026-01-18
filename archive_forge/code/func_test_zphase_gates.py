from typing import Dict, List, Tuple
from cirq.ops.fsim_gate import PhasedFSimGate
import numpy as np
import pytest
import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
from cirq_google.devices.google_noise_properties import (
def test_zphase_gates():
    q0 = cirq.LineQubit(0)
    props = sample_noise_properties([q0], [])
    model = NoiseModelFromGoogleNoiseProperties(props)
    circuit = cirq.Circuit(cirq.Z(q0) ** 0.3)
    noisy_circuit = circuit.with_noise(model)
    assert noisy_circuit == circuit