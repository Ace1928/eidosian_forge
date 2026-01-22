from __future__ import annotations
from collections.abc import Callable
from numpy import pi
from qiskit.circuit import QuantumCircuit, Parameter
from qiskit.circuit.library.standard_gates import RZGate
from .two_local import TwoLocal
Return the parameter bounds.

        Returns:
            The parameter bounds.
        