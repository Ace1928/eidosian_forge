import numpy as np
import pytest
import sympy
import cirq
import cirq_google.api.v1.programs as programs
from cirq_google.api.v1 import operations_pb2
def test_protobuf_round_trip():
    qubits = cirq.GridQubit.rect(1, 5)
    circuit = cirq.Circuit([cirq.X(q) ** 0.5 for q in qubits], [cirq.CZ(q, q2) for q in [cirq.GridQubit(0, 0)] for q, q2 in zip(qubits, qubits) if q != q2])
    protos = list(programs.circuit_as_schedule_to_protos(circuit))
    s2 = programs.circuit_from_schedule_from_protos(protos)
    assert s2 == circuit