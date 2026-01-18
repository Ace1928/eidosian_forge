from math import exp
import pytest
from google.protobuf.text_format import Merge
import cirq
from cirq.testing import assert_equivalent_op_tree
import cirq_google
from cirq_google.api import v2
from cirq_google.experimental.noise_models import simple_noise_from_calibration_metrics
def test_per_qubit_readout_error_from_data():
    calibration = cirq_google.Calibration(_CALIBRATION_DATA)
    noise_model = simple_noise_from_calibration_metrics(calibration=calibration, readout_error_noise=True)
    qubits = [cirq.GridQubit(0, 0), cirq.GridQubit(0, 1), cirq.GridQubit(1, 0)]
    program = cirq.Circuit(cirq.Moment([cirq.H(qubits[0])]), cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]), cirq.Moment([cirq.CNOT(qubits[0], qubits[2])]), cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1'), cirq.measure(qubits[2], key='q2')]))
    noisy_circuit = cirq.Circuit(noise_model.noisy_moments(program, qubits))
    expected_program = cirq.Circuit(cirq.Moment([cirq.H(qubits[0])]), cirq.Moment([cirq.CNOT(qubits[0], qubits[1])]), cirq.Moment([cirq.CNOT(qubits[0], qubits[2])]), cirq.Moment([cirq.BitFlipChannel(0.004).on(qubits[0]), cirq.BitFlipChannel(0.005).on(qubits[1]), cirq.BitFlipChannel(0.006).on(qubits[2])]), cirq.Moment([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1'), cirq.measure(qubits[2], key='q2')]))
    assert_equivalent_op_tree(expected_program, noisy_circuit)