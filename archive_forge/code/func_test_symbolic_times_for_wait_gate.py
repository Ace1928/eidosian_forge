import numpy as np
import pytest
import sympy
import cirq
from cirq.devices.noise_utils import PHYSICAL_GATE_TAG
from cirq.devices.thermal_noise_model import (
def test_symbolic_times_for_wait_gate():
    q0 = cirq.LineQubit(0)
    gate_durations = {cirq.ZPowGate: 25.0}
    heat_rate_GHz = {q0: 1e-05}
    cool_rate_GHz = {q0: 0.0001}
    model = ThermalNoiseModel(qubits={q0}, gate_durations_ns=gate_durations, heat_rate_GHz=heat_rate_GHz, cool_rate_GHz=cool_rate_GHz, dephase_rate_GHz=None, require_physical_tag=False, skip_measurements=True)
    moment = cirq.Moment(cirq.wait(q0, nanos=sympy.Symbol('t')))
    with pytest.raises(ValueError, match='Symbolic'):
        _ = model.noisy_moment(moment, system_qubits=[q0])