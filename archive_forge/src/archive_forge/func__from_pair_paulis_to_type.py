import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford, Pauli
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
def _from_pair_paulis_to_type(pauli_x, pauli_z, qubit):
    """Converts a pair of Paulis pauli_x and pauli_z into a type"""
    type_x = [pauli_x.z[qubit], pauli_x.x[qubit]]
    type_z = [pauli_z.z[qubit], pauli_z.x[qubit]]
    return [type_x, type_z]