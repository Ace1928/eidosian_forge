from __future__ import annotations
import typing
from numbers import Number
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterExpression
from qiskit.circuit.exceptions import CircuitError
from qiskit.quantum_info.operators.predicates import matrix_equal
from qiskit.quantum_info.operators.predicates import is_hermitian_matrix
from .generalized_gates.unitary import UnitaryGate
Hamiltonian parameter has to be an ndarray, operator or float.