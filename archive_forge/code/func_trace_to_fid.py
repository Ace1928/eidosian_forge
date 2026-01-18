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
def trace_to_fid(trace):
    """Average gate fidelity is

    .. math::

        \\bar{F} = \\frac{d + |\\mathrm{Tr} (U_\\text{target} \\cdot U^{\\dag})|^2}{d(d+1)}

    M. Horodecki, P. Horodecki and R. Horodecki, PRA 60, 1888 (1999)"""
    return (4 + abs(trace) ** 2) / 20