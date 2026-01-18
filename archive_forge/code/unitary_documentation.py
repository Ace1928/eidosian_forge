from __future__ import annotations
import typing
import numpy
from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.annotated_operation import AnnotatedOperation, ControlModifier
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.exceptions import CircuitError
from qiskit.circuit._utils import _compute_control_matrix
from qiskit.circuit.library.standard_gates.u import UGate
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer
from qiskit.synthesis.two_qubit.two_qubit_decompose import two_qubit_cnot_decompose
from .isometry import Isometry
Unitary gate parameter has to be an ndarray.