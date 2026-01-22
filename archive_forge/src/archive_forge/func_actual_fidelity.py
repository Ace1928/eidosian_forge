from __future__ import annotations
import cmath
import math
import io
import base64
import warnings
from typing import ClassVar, Optional, Type
import logging
import numpy as np
from qiskit.circuit import QuantumRegister, QuantumCircuit, Gate
from qiskit.circuit.library.standard_gates import CXGate, RXGate, RYGate, RZGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info.operators import Operator
from qiskit.synthesis.two_qubit.weyl import transform_to_magic_basis
from qiskit.synthesis.one_qubit.one_qubit_decompose import (
from qiskit._accelerate import two_qubit_decompose
def actual_fidelity(self, **kwargs) -> float:
    """Calculates the actual fidelity of the decomposed circuit to the input unitary."""
    circ = self.circuit(**kwargs)
    trace = np.trace(Operator(circ).data.T.conj() @ self.unitary_matrix)
    return trace_to_fid(trace)