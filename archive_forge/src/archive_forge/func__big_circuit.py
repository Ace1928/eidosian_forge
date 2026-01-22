import pytest
import cirq
import cirq_google as cg
import cirq_google.engine.engine_validator as engine_validator
def _big_circuit(num_cycles: int) -> cirq.Circuit:
    qubits = cirq.GridQubit.rect(6, 6)
    moment_1q = cirq.Moment([cirq.X(q) for q in qubits])
    moment_2q = cirq.Moment([cirq.CZ(cirq.GridQubit(row, col), cirq.GridQubit(row, col + 1)) for row in range(6) for col in [0, 2, 4]])
    c = cirq.Circuit()
    for _ in range(num_cycles):
        c.append(moment_1q)
        c.append(moment_2q)
    return c