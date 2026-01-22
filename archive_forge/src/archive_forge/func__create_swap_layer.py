from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from .permutation_utils import _inverse_pattern
def _create_swap_layer(qc, pattern, starting_point):
    """Implements a single swap layer, consisting of conditional swaps between each
    neighboring couple. The starting_point is the first qubit to use (either 0 or 1
    for even or odd layers respectively). Mutates both the quantum circuit ``qc``
    and the permutation pattern ``pattern``.
    """
    num_qubits = len(pattern)
    for j in range(starting_point, num_qubits - 1, 2):
        if pattern[j] > pattern[j + 1]:
            qc.swap(j, j + 1)
            pattern[j], pattern[j + 1] = (pattern[j + 1], pattern[j])