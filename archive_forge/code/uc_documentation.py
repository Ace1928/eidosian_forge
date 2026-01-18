from __future__ import annotations
import cmath
import math
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.library.standard_gates.h import HGate
from qiskit.quantum_info.operators.predicates import is_unitary_matrix
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from qiskit.exceptions import QiskitError
from qiskit.synthesis.one_qubit.one_qubit_decompose import OneQubitEulerDecomposer
from .diagonal import Diagonal
Uniformly controlled gate parameter has to be an ndarray.