from __future__ import annotations
from typing import List, Optional
import numpy as np
from qiskit.circuit import QuantumRegister, AncillaRegister, QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
from .functional_pauli_rotations import FunctionalPauliRotations
from .polynomial_pauli_rotations import PolynomialPauliRotations
from .integer_comparator import IntegerComparator
If not already built, build the circuit.