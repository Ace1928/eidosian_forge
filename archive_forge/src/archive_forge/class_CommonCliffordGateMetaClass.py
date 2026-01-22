from typing import Any, Dict, List, Optional, Sequence, Tuple, TYPE_CHECKING, Union
import numpy as np
from cirq import protocols, value, linalg, qis
from cirq._import import LazyLoader
from cirq.ops import common_gates, named_qubit, raw_types, pauli_gates, phased_x_z_gate
from cirq.ops.pauli_gates import Pauli
from cirq.type_workarounds import NotImplementedType
class CommonCliffordGateMetaClass(value.ABCMetaImplementAnyOneOf):
    """A metaclass used to lazy initialize several common Clifford Gate as class attributes."""

    @property
    def all_single_qubit_cliffords(cls) -> Sequence['cirq.SingleQubitCliffordGate']:
        """All 24 single-qubit Clifford gates."""
        if not hasattr(cls, '_all_single_qubit_cliffords'):
            pX = (pauli_gates.X, False)
            mX = (pauli_gates.X, True)
            pY = (pauli_gates.Y, False)
            mY = (pauli_gates.Y, True)
            pZ = (pauli_gates.Z, False)
            mZ = (pauli_gates.Z, True)

            def from_xz(x_to, z_to):
                return SingleQubitCliffordGate.from_clifford_tableau(_to_clifford_tableau(x_to=x_to, z_to=z_to))
            cls._all_single_qubit_cliffords = (from_xz(x_to=pX, z_to=pZ), from_xz(x_to=pX, z_to=mZ), from_xz(x_to=mX, z_to=mZ), from_xz(x_to=mX, z_to=pZ), from_xz(x_to=pX, z_to=mY), from_xz(x_to=mZ, z_to=pX), from_xz(x_to=pY, z_to=pZ), from_xz(x_to=pX, z_to=pY), from_xz(x_to=pZ, z_to=mX), from_xz(x_to=mY, z_to=pZ), from_xz(x_to=pZ, z_to=pX), from_xz(x_to=pY, z_to=mZ), from_xz(x_to=mX, z_to=pY), from_xz(x_to=mZ, z_to=mX), from_xz(x_to=mY, z_to=mZ), from_xz(x_to=mX, z_to=mY), from_xz(x_to=pY, z_to=pX), from_xz(x_to=mZ, z_to=mY), from_xz(x_to=pZ, z_to=mY), from_xz(x_to=mY, z_to=mX), from_xz(x_to=mZ, z_to=pY), from_xz(x_to=mY, z_to=pX), from_xz(x_to=pY, z_to=mX), from_xz(x_to=pZ, z_to=pY))
        return cls._all_single_qubit_cliffords

    @property
    def I(cls) -> 'cirq.SingleQubitCliffordGate':
        return cls.all_single_qubit_cliffords[0]

    @property
    def X(cls) -> 'cirq.SingleQubitCliffordGate':
        return cls.all_single_qubit_cliffords[1]

    @property
    def Y(cls) -> 'cirq.SingleQubitCliffordGate':
        return cls.all_single_qubit_cliffords[2]

    @property
    def Z(cls) -> 'cirq.SingleQubitCliffordGate':
        return cls.all_single_qubit_cliffords[3]

    @property
    def H(cls) -> 'cirq.SingleQubitCliffordGate':
        return cls.all_single_qubit_cliffords[10]

    @property
    def S(cls) -> 'cirq.SingleQubitCliffordGate':
        return cls.all_single_qubit_cliffords[6]

    @property
    def CNOT(cls) -> 'cirq.CliffordGate':
        if not hasattr(cls, '_CNOT'):
            t = qis.CliffordTableau(num_qubits=2)
            t.xs = np.array([[1, 1], [0, 1], [0, 0], [0, 0]])
            t.zs = np.array([[0, 0], [0, 0], [1, 0], [1, 1]])
            cls._CNOT = CliffordGate.from_clifford_tableau(t)
        return cls._CNOT

    @property
    def CZ(cls) -> 'cirq.CliffordGate':
        if not hasattr(cls, '_CZ'):
            t = qis.CliffordTableau(num_qubits=2)
            t.xs = np.array([[1, 0], [0, 1], [0, 0], [0, 0]])
            t.zs = np.array([[0, 1], [1, 0], [1, 0], [0, 1]])
            cls._CZ = CliffordGate.from_clifford_tableau(t)
        return cls._CZ

    @property
    def SWAP(cls) -> 'cirq.CliffordGate':
        if not hasattr(cls, '_SWAP'):
            t = qis.CliffordTableau(num_qubits=2)
            t.xs = np.array([[0, 1], [1, 0], [0, 0], [0, 0]])
            t.zs = np.array([[0, 0], [0, 0], [0, 1], [1, 0]])
            cls._SWAP = CliffordGate.from_clifford_tableau(t)
        return cls._SWAP

    @property
    def X_sqrt(cls) -> 'cirq.SingleQubitCliffordGate':
        return cls.all_single_qubit_cliffords[4]

    @property
    def X_nsqrt(cls) -> 'cirq.SingleQubitCliffordGate':
        return cls.all_single_qubit_cliffords[7]

    @property
    def Y_sqrt(cls) -> 'cirq.SingleQubitCliffordGate':
        return cls.all_single_qubit_cliffords[5]

    @property
    def Y_nsqrt(cls) -> 'cirq.SingleQubitCliffordGate':
        return cls.all_single_qubit_cliffords[8]

    @property
    def Z_sqrt(cls) -> 'cirq.SingleQubitCliffordGate':
        return cls.all_single_qubit_cliffords[6]

    @property
    def Z_nsqrt(cls) -> 'cirq.SingleQubitCliffordGate':
        return cls.all_single_qubit_cliffords[9]