from __future__ import annotations
import warnings
import collections
import numpy as np
import qiskit.circuit.library.standard_gates as gates
from qiskit.circuit import Gate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.utils import optionals
from .gate_sequence import GateSequence
def _process_node(node: Node, basis: list[str], sequences: list[GateSequence]):
    inverse_last = _1q_inverses[node.labels[-1]] if node.labels else None
    for label in basis:
        if label == inverse_last:
            continue
        sequence = node.sequence.copy()
        sequence.append(_1q_gates[label])
        if _check_candidate(sequence, sequences):
            sequences.append(sequence)
            node.children.append(Node(node.labels + (label,), sequence, []))
    return node.children