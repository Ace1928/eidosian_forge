import pytest
import cirq
@pytest.mark.parametrize('rot, obs, out', [(cirq.I, cirq.DensePauliString('Z', coefficient=+1), 0), (cirq.I, cirq.DensePauliString('Z', coefficient=-1), 1), (cirq.Y ** 0.5, cirq.DensePauliString('X', coefficient=+1), 0), (cirq.Y ** 0.5, cirq.DensePauliString('X', coefficient=-1), 1), (cirq.X ** (-0.5), cirq.DensePauliString('Y', coefficient=+1), 0), (cirq.X ** (-0.5), cirq.DensePauliString('Y', coefficient=-1), 1)])
def test_pauli_measurement_gate_samples(rot, obs, out):
    q = cirq.NamedQubit('q')
    c = cirq.Circuit(rot(q), cirq.PauliMeasurementGate(obs, key='out').on(q))
    assert cirq.Simulator().sample(c)['out'][0] == out