import duet
import cirq
def test_pauli_string_sample_collector_extra_qubit_z():
    a, b = cirq.LineQubit.range(2)
    p = cirq.PauliSumCollector(circuit=cirq.Circuit(cirq.H(a)), observable=3 * cirq.Z(b), samples_per_term=100)
    p.collect(sampler=cirq.Simulator())
    assert p.estimated_energy() == 3