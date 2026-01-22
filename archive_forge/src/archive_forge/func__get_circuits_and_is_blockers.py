import itertools
import random
import pytest
import networkx
import cirq
def _get_circuits_and_is_blockers():
    qubits = cirq.LineQubit.range(10)
    circuits = [cirq.testing.random_circuit(qubits, 10, 0.5) for _ in range(1)]
    edges = [set(qubit_pair) for qubit_pair in itertools.combinations(qubits, 2) if random.random() > 0.5]
    not_on_edge = lambda op: len(op.qubits) > 1 and set(op.qubits) not in edges
    is_blockers = [lambda op: False, not_on_edge]
    return itertools.product(circuits, is_blockers)