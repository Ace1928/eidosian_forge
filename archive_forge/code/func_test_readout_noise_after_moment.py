import cirq
import cirq.contrib.noise_models as ccn
from cirq import ops
from cirq.testing import assert_equivalent_op_tree
def test_readout_noise_after_moment():
    program = cirq.Circuit()
    qubits = cirq.LineQubit.range(3)
    program.append([cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1]), cirq.CNOT(qubits[1], qubits[2])])
    program.append([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1'), cirq.measure(qubits[2], key='q2')], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    depol_noise = ccn.DepolarizingNoiseModel(depol_prob=0.01)
    readout_noise = ccn.ReadoutNoiseModel(bitflip_prob=0.05)
    noisy_circuit = cirq.Circuit(depol_noise.noisy_moments(program, qubits))
    noisy_circuit = cirq.Circuit(readout_noise.noisy_moments(noisy_circuit, qubits))
    true_noisy_program = cirq.Circuit()
    true_noisy_program.append([cirq.H(qubits[0])])
    true_noisy_program.append([cirq.DepolarizingChannel(0.01).on(q).with_tags(ops.VirtualTag()) for q in qubits], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    true_noisy_program.append([cirq.CNOT(qubits[0], qubits[1])])
    true_noisy_program.append([cirq.DepolarizingChannel(0.01).on(q).with_tags(ops.VirtualTag()) for q in qubits], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    true_noisy_program.append([cirq.CNOT(qubits[1], qubits[2])])
    true_noisy_program.append([cirq.DepolarizingChannel(0.01).on(q).with_tags(ops.VirtualTag()) for q in qubits], strategy=cirq.InsertStrategy.NEW_THEN_INLINE)
    true_noisy_program.append([cirq.BitFlipChannel(0.05).on(q).with_tags(ops.VirtualTag()) for q in qubits])
    true_noisy_program.append([cirq.measure(qubits[0], key='q0'), cirq.measure(qubits[1], key='q1'), cirq.measure(qubits[2], key='q2')])
    assert_equivalent_op_tree(true_noisy_program, noisy_circuit)