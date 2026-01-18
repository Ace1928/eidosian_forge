from __future__ import annotations
from collections.abc import Collection
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators.op_shape import OpShape
from qiskit.quantum_info.operators.operator import Operator
from qiskit.quantum_info.operators.symplectic import Clifford, Pauli, PauliList
from qiskit.quantum_info.operators.symplectic.clifford_circuits import _append_x
from qiskit.quantum_info.states.quantum_state import QuantumState
from qiskit.circuit import QuantumCircuit, Instruction
Recursive helper function for calculating the probabilities