import itertools
import uuid
from typing import Iterable
from qiskit.circuit import (
from qiskit.circuit.classical import expr
def _circuit_to_dag(circuit, node_qargs, node_cargs, bit_indices):
    """Get a :class:`.DAGCircuit` of the given :class:`.QuantumCircuit`.  The bits in the output
    will be ordered in a canonical order based on their indices in the outer DAG, as defined by the
    ``bit_indices`` mapping and the ``node_{q,c}args`` arguments."""
    from qiskit.converters import circuit_to_dag

    def sort_key(bits):
        outer, _inner = bits
        return bit_indices[outer]
    return circuit_to_dag(circuit, copy_operations=False, qubit_order=[inner for _outer, inner in sorted(zip(node_qargs, circuit.qubits), key=sort_key)], clbit_order=[inner for _outer, inner in sorted(zip(node_cargs, circuit.clbits), key=sort_key)])