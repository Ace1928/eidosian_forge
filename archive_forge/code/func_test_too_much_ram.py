import functools
import operator
import numpy as np
import pytest
import cirq
import cirq.contrib.quimb as ccq
def test_too_much_ram():
    qubits = cirq.LineQubit.range(30)
    circuit = cirq.testing.random_circuit(qubits=qubits, n_moments=20, op_density=0.8)
    op = functools.reduce(operator.mul, [cirq.Z(q) for q in qubits], 1)
    with pytest.raises(MemoryError) as e:
        ccq.tensor_expectation_value(circuit=circuit, pauli_string=op)
    assert e.match('.*too much RAM!.*')