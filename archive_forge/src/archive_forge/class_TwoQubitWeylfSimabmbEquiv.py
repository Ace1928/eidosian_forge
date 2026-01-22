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
class TwoQubitWeylfSimabmbEquiv(TwoQubitWeylDecomposition):
    """:math:`U \\sim U_d(\\alpha, \\beta, -\\beta), \\alpha \\geq \\beta \\geq 0`

    This gate binds 5 parameters, we make it canonical by setting:
    :math:`K2_l = Ry(\\theta_l)Rx(\\lambda_l)`.
    """
    _default_1q_basis = 'XYX'

    def specialize(self):
        self.b = (self.b - self.c) / 2
        self.c = -self.b
        k2ltheta, k2lphi, k2llambda, k2lphase = _oneq_xyx.angles_and_phase(self.K2l)
        self.global_phase += k2lphase
        self.K1r = self.K1r @ _ipz @ np.asarray(RXGate(k2lphi)) @ _ipz
        self.K1l = self.K1l @ np.asarray(RXGate(k2lphi))
        self.K2l = np.asarray(RYGate(k2ltheta)) @ np.asarray(RXGate(k2llambda))
        self.K2r = _ipz @ np.asarray(RXGate(-k2lphi)) @ _ipz @ self.K2r