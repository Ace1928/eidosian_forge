import duet
import cirq
def test_pauli_string_sample_collector_identity():
    p = cirq.PauliSumCollector(circuit=cirq.Circuit(), observable=cirq.PauliSum() + 2j, samples_per_term=100)
    p.collect(sampler=cirq.Simulator())
    assert p.estimated_energy() == 2j