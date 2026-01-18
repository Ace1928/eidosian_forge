from __future__ import annotations
from typing import Callable
import numpy as np
from numpy.polynomial.chebyshev import Chebyshev
from qiskit.circuit import QuantumRegister, AncillaRegister
from qiskit.circuit.library.blueprintcircuit import BlueprintCircuit
from qiskit.circuit.exceptions import CircuitError
from .piecewise_polynomial_pauli_rotations import PiecewisePolynomialPauliRotations
Build the circuit if not already build. The operation is considered successful
        when q_objective is :math:`|1>`