import cirq
import cirq_ionq as ionq
import pytest
from cirq_ionq.ionq_gateset_test import VALID_GATES
def test_validate_moment_valid():
    moment = cirq.Moment()
    q = 0
    all_qubits = []
    for gate in VALID_GATES:
        qubits = cirq.LineQubit.range(q, q + gate.num_qubits())
        all_qubits.extend(qubits)
        moment += [gate(*qubits)]
        q += gate.num_qubits()
    device = ionq.IonQAPIDevice(len(all_qubits))
    device.validate_moment(moment)