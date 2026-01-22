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
class TwoQubitWeylControlledEquiv(TwoQubitWeylDecomposition):
    """:math:`U \\sim U_d(\\alpha, 0, 0) \\sim \\text{Ctrl-U}`
    This gate binds 4 parameters, we make it canonical by setting:
        :math:`K2_l = Ry(\\theta_l) Rx(\\lambda_l)` ,
        :math:`K2_r = Ry(\\theta_r) Rx(\\lambda_r)` .
    """
    _default_1q_basis = 'XYX'

    def specialize(self):
        self.b = self.c = 0
        k2ltheta, k2lphi, k2llambda, k2lphase = _oneq_xyx.angles_and_phase(self.K2l)
        k2rtheta, k2rphi, k2rlambda, k2rphase = _oneq_xyx.angles_and_phase(self.K2r)
        self.global_phase += k2lphase + k2rphase
        self.K1l = self.K1l @ np.asarray(RXGate(k2lphi))
        self.K1r = self.K1r @ np.asarray(RXGate(k2rphi))
        self.K2l = np.asarray(RYGate(k2ltheta)) @ np.asarray(RXGate(k2llambda))
        self.K2r = np.asarray(RYGate(k2rtheta)) @ np.asarray(RXGate(k2rlambda))