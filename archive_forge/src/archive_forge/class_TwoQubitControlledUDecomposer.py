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
class TwoQubitControlledUDecomposer:
    """Decompose two-qubit unitary in terms of a desired
    :math:`U \\sim U_d(\\alpha, 0, 0) \\sim \\text{Ctrl-U}`
    gate that is locally equivalent to an :class:`.RXXGate`."""

    def __init__(self, rxx_equivalent_gate: Type[Gate]):
        """Initialize the KAK decomposition.

        Args:
            rxx_equivalent_gate: Gate that is locally equivalent to an :class:`.RXXGate`:
            :math:`U \\sim U_d(\\alpha, 0, 0) \\sim \\text{Ctrl-U}` gate.
        Raises:
            QiskitError: If the gate is not locally equivalent to an :class:`.RXXGate`.
        """
        atol = DEFAULT_ATOL
        scales, test_angles, scale = ([], [0.2, 0.3, np.pi / 2], None)
        for test_angle in test_angles:
            try:
                rxx_equivalent_gate(test_angle, label='foo')
            except TypeError as _:
                raise QiskitError('Equivalent gate needs to take exactly 1 angle parameter.') from _
            decomp = TwoQubitWeylDecomposition(rxx_equivalent_gate(test_angle))
            circ = QuantumCircuit(2)
            circ.rxx(test_angle, 0, 1)
            decomposer_rxx = TwoQubitWeylControlledEquiv(Operator(circ).data)
            circ = QuantumCircuit(2)
            circ.append(rxx_equivalent_gate(test_angle), qargs=[0, 1])
            decomposer_equiv = TwoQubitWeylControlledEquiv(Operator(circ).data)
            scale = decomposer_rxx.a / decomposer_equiv.a
            if not isinstance(decomp, TwoQubitWeylControlledEquiv) or abs(decomp.a * 2 - test_angle / scale) > atol:
                raise QiskitError(f'{rxx_equivalent_gate.__name__} is not equivalent to an RXXGate.')
            scales.append(scale)
        if not np.allclose(scales, [scale] * len(test_angles)):
            raise QiskitError(f'Cannot initialize {self.__class__.__name__}: with gate {rxx_equivalent_gate}. Inconsistent scaling parameters in checks.')
        self.scale = scales[0]
        self.rxx_equivalent_gate = rxx_equivalent_gate

    def __call__(self, unitary, *, atol=DEFAULT_ATOL) -> QuantumCircuit:
        """Returns the Weyl decomposition in circuit form.

        Note: atol ist passed to OneQubitEulerDecomposer.
        """
        self.decomposer = TwoQubitWeylDecomposition(unitary)
        oneq_decompose = OneQubitEulerDecomposer('ZYZ')
        c1l, c1r, c2l, c2r = (oneq_decompose(k, atol=atol) for k in (self.decomposer.K1l, self.decomposer.K1r, self.decomposer.K2l, self.decomposer.K2r))
        circ = QuantumCircuit(2, global_phase=self.decomposer.global_phase)
        circ.compose(c2r, [0], inplace=True)
        circ.compose(c2l, [1], inplace=True)
        self._weyl_gate(circ)
        circ.compose(c1r, [0], inplace=True)
        circ.compose(c1l, [1], inplace=True)
        return circ

    def _to_rxx_gate(self, angle: float) -> QuantumCircuit:
        """
        Takes an angle and returns the circuit equivalent to an RXXGate with the
        RXX equivalent gate as the two-qubit unitary.

        Args:
            angle: Rotation angle (in this case one of the Weyl parameters a, b, or c)

        Returns:
            Circuit: Circuit equivalent to an RXXGate.

        Raises:
            QiskitError: If the circuit is not equivalent to an RXXGate.
        """
        circ = QuantumCircuit(2)
        circ.append(self.rxx_equivalent_gate(self.scale * angle), qargs=[0, 1])
        decomposer_inv = TwoQubitWeylControlledEquiv(Operator(circ).data)
        oneq_decompose = OneQubitEulerDecomposer('ZYZ')
        rxx_circ = QuantumCircuit(2, global_phase=-decomposer_inv.global_phase)
        rxx_circ.compose(oneq_decompose(decomposer_inv.K2r).inverse(), inplace=True, qubits=[0])
        rxx_circ.compose(oneq_decompose(decomposer_inv.K2l).inverse(), inplace=True, qubits=[1])
        rxx_circ.compose(circ, inplace=True)
        rxx_circ.compose(oneq_decompose(decomposer_inv.K1r).inverse(), inplace=True, qubits=[0])
        rxx_circ.compose(oneq_decompose(decomposer_inv.K1l).inverse(), inplace=True, qubits=[1])
        return rxx_circ

    def _weyl_gate(self, circ: QuantumCircuit, atol=1e-13):
        """Appends U_d(a, b, c) to the circuit."""
        circ_rxx = self._to_rxx_gate(-2 * self.decomposer.a)
        circ.compose(circ_rxx, inplace=True)
        if abs(self.decomposer.b) > atol:
            circ_ryy = QuantumCircuit(2)
            circ_ryy.sdg(0)
            circ_ryy.sdg(1)
            circ_ryy.compose(self._to_rxx_gate(-2 * self.decomposer.b), inplace=True)
            circ_ryy.s(0)
            circ_ryy.s(1)
            circ.compose(circ_ryy, inplace=True)
        if abs(self.decomposer.c) > atol:
            gamma, invert = (-2 * self.decomposer.c, False)
            if gamma > 0:
                gamma *= -1
                invert = True
            circ_rzz = QuantumCircuit(2)
            circ_rzz.h(0)
            circ_rzz.h(1)
            circ_rzz.compose(self._to_rxx_gate(gamma), inplace=True)
            circ_rzz.h(0)
            circ_rzz.h(1)
            if invert:
                circ.compose(circ_rzz.inverse(), inplace=True)
            else:
                circ.compose(circ_rzz, inplace=True)
        return circ