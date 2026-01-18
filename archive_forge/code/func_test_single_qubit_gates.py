from typing import Dict, List, Tuple
from cirq.ops.fsim_gate import PhasedFSimGate
import numpy as np
import pytest
import cirq, cirq_google
from cirq.devices.noise_utils import OpIdentifier, PHYSICAL_GATE_TAG
from cirq_google.devices.google_noise_properties import (
@pytest.mark.parametrize('op', [(cirq.Z(cirq.LineQubit(0)) ** 0.3).with_tags(cirq_google.PhysicalZTag), cirq.PhasedXZGate(x_exponent=0.8, z_exponent=0.2, axis_phase_exponent=0.1).on(cirq.LineQubit(0))])
def test_single_qubit_gates(op):
    q0 = cirq.LineQubit(0)
    props = sample_noise_properties([q0], [])
    model = NoiseModelFromGoogleNoiseProperties(props)
    circuit = cirq.Circuit(op)
    noisy_circuit = circuit.with_noise(model)
    assert len(noisy_circuit.moments) == 3
    assert len(noisy_circuit.moments[0].operations) == 1
    assert noisy_circuit.moments[0].operations[0] == op.with_tags(PHYSICAL_GATE_TAG)
    assert len(noisy_circuit.moments[1].operations) == 1
    depol_op = noisy_circuit.moments[1].operations[0]
    assert isinstance(depol_op.gate, cirq.DepolarizingChannel)
    assert np.isclose(depol_op.gate.p, 0.00081252)
    assert len(noisy_circuit.moments[2].operations) == 1
    thermal_op = noisy_circuit.moments[2].operations[0]
    assert isinstance(thermal_op.gate, cirq.KrausChannel)
    thermal_choi = cirq.kraus_to_choi(cirq.kraus(thermal_op))
    assert np.allclose(thermal_choi, [[1, 0, 0, 0.999750031], [0, 0.000249968753, 0, 0], [0, 0, 0, 0], [0.999750031, 0, 0, 0.999750031]])
    depol_pauli_err = 1 - cirq.qis.measures.entanglement_fidelity(depol_op)
    thermal_pauli_err = 1 - cirq.qis.measures.entanglement_fidelity(thermal_op)
    total_err = depol_pauli_err + thermal_pauli_err
    assert np.isclose(total_err, SINGLE_QUBIT_ERROR)