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
def _closest_partial_swap(a, b, c) -> float:
    """A good approximation to the best value x to get the minimum
    trace distance for :math:`U_d(x, x, x)` from :math:`U_d(a, b, c)`.
    """
    m = (a + b + c) / 3
    am, bm, cm = (a - m, b - m, c - m)
    ab, bc, ca = (a - b, b - c, c - a)
    return m + am * bm * cm * (6 + ab * ab + bc * bc + ca * ca) / 18