import dataclasses
import cirq
import cirq_google
import pytest
from cirq_google import (
def test_quantum_executable_inputs():
    qubits = cirq.GridQubit.rect(2, 3)
    spec = _get_example_spec(name='example-program')
    circuit = _get_random_circuit(qubits)
    measurement = BitstringsMeasurement(n_repetitions=10)
    params1 = {'theta': 0.2}
    params2 = cirq.ParamResolver({'theta': 0.2})
    params3 = [('theta', 0.2)]
    params4 = (('theta', 0.2),)
    exes = [QuantumExecutable(spec=spec, circuit=circuit, measurement=measurement, params=p) for p in [params1, params2, params3, params4]]
    for exe in exes:
        assert exe == exes[0]
    with pytest.raises(ValueError):
        _ = QuantumExecutable(spec=spec, circuit=circuit, measurement=measurement, params='theta=0.2')
    with pytest.raises(TypeError):
        _ = QuantumExecutable(spec={'name': 'main'}, circuit=circuit, measurement=measurement)