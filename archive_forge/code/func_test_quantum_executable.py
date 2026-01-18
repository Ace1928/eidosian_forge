import dataclasses
import cirq
import cirq_google
import pytest
from cirq_google import (
def test_quantum_executable(tmpdir):
    qubits = cirq.GridQubit.rect(2, 2)
    exe = QuantumExecutable(spec=_get_example_spec(name='example-program'), circuit=_get_random_circuit(qubits), measurement=BitstringsMeasurement(n_repetitions=10))
    assert isinstance(exe.circuit, cirq.FrozenCircuit)
    assert hash(exe) is not None
    assert hash(dataclasses.astuple(exe)) is not None
    assert hash(dataclasses.astuple(exe)) == exe._hash
    prog2 = QuantumExecutable(spec=_get_example_spec(name='example-program'), circuit=_get_random_circuit(qubits), measurement=BitstringsMeasurement(n_repetitions=10))
    assert exe == prog2
    assert hash(exe) == hash(prog2)
    prog3 = QuantumExecutable(spec=_get_example_spec(name='example-program'), circuit=_get_random_circuit(qubits), measurement=BitstringsMeasurement(n_repetitions=20))
    assert exe != prog3
    assert hash(exe) != hash(prog3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        prog3.measurement.n_repetitions = 10
    cirq.to_json(exe, f'{tmpdir}/exe.json')
    exe_reconstructed = cirq.read_json(f'{tmpdir}/exe.json')
    assert exe == exe_reconstructed
    assert str(exe) == "QuantumExecutable(spec=cirq_google.KeyValueExecutableSpec(executable_family='cirq_google.algo_benchmarks.example', key_value_pairs=(('name', 'example-program'),)))"
    cirq.testing.assert_equivalent_repr(exe, global_vals={'cirq_google': cirq_google})