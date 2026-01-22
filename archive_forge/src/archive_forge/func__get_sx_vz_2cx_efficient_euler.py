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
def _get_sx_vz_2cx_efficient_euler(self, decomposition, target_decomposed):
    """
        Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT gates assuming
        two CNOT gates are needed.

        This first decomposes each unitary from the KAK decomposition into ZXZ on the source
        qubit of the CNOTs and XZX on the targets in order to commute operators to beginning and
        end of decomposition. The beginning and ending single qubit gates are then
        collapsed and re-decomposed with the single qubit decomposer. This last step could be avoided
        if performance is a concern.
        """
    best_nbasis = 2
    num_1q_uni = len(decomposition)
    euler_q0 = np.empty((num_1q_uni // 2, 3), dtype=float)
    euler_q1 = np.empty((num_1q_uni // 2, 3), dtype=float)
    global_phase = 0.0
    zxz_decomposer = OneQubitEulerDecomposer('ZXZ')
    for iqubit, decomp in enumerate(decomposition[0::2]):
        euler_angles = zxz_decomposer.angles_and_phase(decomp)
        euler_q0[iqubit, [1, 2, 0]] = euler_angles[:3]
        global_phase += euler_angles[3]
    xzx_decomposer = OneQubitEulerDecomposer('XZX')
    for iqubit, decomp in enumerate(decomposition[1::2]):
        euler_angles = xzx_decomposer.angles_and_phase(decomp)
        euler_q1[iqubit, [1, 2, 0]] = euler_angles[:3]
        global_phase += euler_angles[3]
    qc = QuantumCircuit(2)
    qc.global_phase = target_decomposed.global_phase
    qc.global_phase -= best_nbasis * self.basis.global_phase
    qc.global_phase += global_phase
    circ = QuantumCircuit(1)
    circ.rz(euler_q0[0][0], 0)
    circ.rx(euler_q0[0][1], 0)
    circ.rz(euler_q0[0][2] + euler_q0[1][0] + math.pi / 2, 0)
    qceuler = self._decomposer1q(Operator(circ).data)
    qc.compose(qceuler, [0], inplace=True)
    circ = QuantumCircuit(1)
    circ.rx(euler_q1[0][0], 0)
    circ.rz(euler_q1[0][1], 0)
    circ.rx(euler_q1[0][2] + euler_q1[1][0], 0)
    qceuler = self._decomposer1q(Operator(circ).data)
    qc.compose(qceuler, [1], inplace=True)
    qc.cx(0, 1)
    qc.sx(0)
    qc.rz(euler_q0[1][1] - math.pi, 0)
    qc.sx(0)
    qc.rz(euler_q1[1][1], 1)
    qc.global_phase += math.pi / 2
    qc.cx(0, 1)
    circ = QuantumCircuit(1)
    circ.rz(euler_q0[1][2] + euler_q0[2][0] + math.pi / 2, 0)
    circ.rx(euler_q0[2][1], 0)
    circ.rz(euler_q0[2][2], 0)
    qceuler = self._decomposer1q(Operator(circ).data)
    qc.compose(qceuler, [0], inplace=True)
    circ = QuantumCircuit(1)
    circ.rx(euler_q1[1][2] + euler_q1[2][0], 0)
    circ.rz(euler_q1[2][1], 0)
    circ.rx(euler_q1[2][2], 0)
    qceuler = self._decomposer1q(Operator(circ).data)
    qc.compose(qceuler, [1], inplace=True)
    return qc