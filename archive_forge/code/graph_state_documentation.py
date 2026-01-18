from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
Create graph state preparation circuit.

        Args:
            adjacency_matrix: input graph as n-by-n list of 0-1 lists

        Raises:
            CircuitError: If adjacency_matrix is not symmetric.

        The circuit prepares a graph state with the given adjacency
        matrix.
        