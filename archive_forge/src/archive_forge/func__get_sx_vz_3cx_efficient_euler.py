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
def _get_sx_vz_3cx_efficient_euler(self, decomposition, target_decomposed):
    """
        Decomposition of SU(4) gate for device with SX, virtual RZ, and CNOT gates assuming
        three CNOT gates are needed.

        This first decomposes each unitary from the KAK decomposition into ZXZ on the source
        qubit of the CNOTs and XZX on the targets in order commute operators to beginning and
        end of decomposition. Inserting Hadamards reverses the direction of the CNOTs and transforms
        a variable Rx -> variable virtual Rz. The beginning and ending single qubit gates are then
        collapsed and re-decomposed with the single qubit decomposer. This last step could be avoided
        if performance is a concern.
        """
    best_nbasis = 3
    num_1q_uni = len(decomposition)
    euler_q0 = np.empty((num_1q_uni // 2, 3), dtype=float)
    euler_q1 = np.empty((num_1q_uni // 2, 3), dtype=float)
    global_phase = 0.0
    atol = 1e-10
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
    x12 = euler_q0[1][2] + euler_q0[2][0]
    x12_isNonZero = not math.isclose(x12, 0, abs_tol=atol)
    x12_isOddMult = None
    x12_isPiMult = math.isclose(math.sin(x12), 0, abs_tol=atol)
    if x12_isPiMult:
        x12_isOddMult = math.isclose(math.cos(x12), -1, abs_tol=atol)
        x12_phase = math.pi * math.cos(x12)
    x02_add = x12 - euler_q0[1][0]
    x12_isHalfPi = math.isclose(x12, math.pi / 2, abs_tol=atol)
    circ = QuantumCircuit(1)
    circ.rz(euler_q0[0][0], 0)
    circ.rx(euler_q0[0][1], 0)
    if x12_isNonZero and x12_isPiMult:
        circ.rz(euler_q0[0][2] - x02_add, 0)
    else:
        circ.rz(euler_q0[0][2] + euler_q0[1][0], 0)
    circ.h(0)
    qceuler = self._decomposer1q(Operator(circ).data)
    qc.compose(qceuler, [0], inplace=True)
    circ = QuantumCircuit(1)
    circ.rx(euler_q1[0][0], 0)
    circ.rz(euler_q1[0][1], 0)
    circ.rx(euler_q1[0][2] + euler_q1[1][0], 0)
    circ.h(0)
    qceuler = self._decomposer1q(Operator(circ).data)
    qc.compose(qceuler, [1], inplace=True)
    qc.cx(1, 0)
    if x12_isPiMult:
        if x12_isNonZero:
            qc.global_phase += x12_phase
        if x12_isNonZero and x12_isOddMult:
            qc.rz(-euler_q0[1][1], 0)
        else:
            qc.rz(euler_q0[1][1], 0)
            qc.global_phase += math.pi
    if x12_isHalfPi:
        qc.sx(0)
        qc.global_phase -= math.pi / 4
    elif x12_isNonZero and (not x12_isPiMult):
        if self.pulse_optimize is None:
            qc.compose(self._decomposer1q(Operator(RXGate(x12)).data), [0], inplace=True)
        else:
            raise QiskitError('possible non-pulse-optimal decomposition encountered')
    if math.isclose(euler_q1[1][1], math.pi / 2, abs_tol=atol):
        qc.sx(1)
        qc.global_phase -= math.pi / 4
    elif self.pulse_optimize is None:
        qc.compose(self._decomposer1q(Operator(RXGate(euler_q1[1][1])).data), [1], inplace=True)
    else:
        raise QiskitError('possible non-pulse-optimal decomposition encountered')
    qc.rz(euler_q1[1][2] + euler_q1[2][0], 1)
    qc.cx(1, 0)
    qc.rz(euler_q0[2][1], 0)
    if math.isclose(euler_q1[2][1], math.pi / 2, abs_tol=atol):
        qc.sx(1)
        qc.global_phase -= math.pi / 4
    elif self.pulse_optimize is None:
        qc.compose(self._decomposer1q(Operator(RXGate(euler_q1[2][1])).data), [1], inplace=True)
    else:
        raise QiskitError('possible non-pulse-optimal decomposition encountered')
    qc.cx(1, 0)
    circ = QuantumCircuit(1)
    circ.h(0)
    circ.rz(euler_q0[2][2] + euler_q0[3][0], 0)
    circ.rx(euler_q0[3][1], 0)
    circ.rz(euler_q0[3][2], 0)
    qceuler = self._decomposer1q(Operator(circ).data)
    qc.compose(qceuler, [0], inplace=True)
    circ = QuantumCircuit(1)
    circ.h(0)
    circ.rx(euler_q1[2][2] + euler_q1[3][0], 0)
    circ.rz(euler_q1[3][1], 0)
    circ.rx(euler_q1[3][2], 0)
    qceuler = self._decomposer1q(Operator(circ).data)
    qc.compose(qceuler, [1], inplace=True)
    if cmath.isclose(target_decomposed.unitary_matrix[0, 0], -Operator(qc).data[0, 0], abs_tol=atol):
        qc.global_phase += math.pi
    return qc