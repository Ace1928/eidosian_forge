import cmath
import math
import numbers
from typing import (
import numpy as np
import sympy
import cirq
from cirq import value, protocols, linalg, qis
from cirq._doc import document
from cirq._import import LazyLoader
from cirq.ops import (
from cirq.type_workarounds import NotImplementedType
class SingleQubitPauliStringGateOperation(gate_operation.GateOperation, PauliString):
    """An operation to represent single qubit pauli gates applied to a qubit.

    Satisfies the contract of both `cirq.GateOperation` and `cirq.PauliString`. Relies
    implicitly on the fact that PauliString({q: X}) compares as equal to
    GateOperation(X, [q]).
    """

    def __init__(self, pauli: pauli_gates.Pauli, qubit: 'cirq.Qid'):
        PauliString.__init__(self, qubit_pauli_map={qubit: pauli})
        gate_operation.GateOperation.__init__(self, cast(raw_types.Gate, pauli), [qubit])

    def with_qubits(self, *new_qubits: 'cirq.Qid') -> 'SingleQubitPauliStringGateOperation':
        if len(new_qubits) != 1:
            raise ValueError('len(new_qubits) != 1')
        return SingleQubitPauliStringGateOperation(cast(pauli_gates.Pauli, self.gate), new_qubits[0])

    @property
    def pauli(self) -> pauli_gates.Pauli:
        return cast(pauli_gates.Pauli, self.gate)

    @property
    def qubit(self) -> raw_types.Qid:
        assert len(self.qubits) == 1
        return self.qubits[0]

    def _as_pauli_string(self) -> PauliString:
        return PauliString(qubit_pauli_map={self.qubit: self.pauli})

    def __mul__(self, other):
        if isinstance(other, SingleQubitPauliStringGateOperation):
            return self._as_pauli_string() * other._as_pauli_string()
        if isinstance(other, (PauliString, complex, float, int)):
            return self._as_pauli_string() * other
        return NotImplemented

    def __rmul__(self, other):
        if isinstance(other, (PauliString, complex, float, int)):
            return other * self._as_pauli_string()
        return NotImplemented

    def __neg__(self):
        return -self._as_pauli_string()

    def _json_dict_(self) -> Dict[str, Any]:
        return protocols.obj_to_dict_helper(self, ['pauli', 'qubit'])

    @classmethod
    def _from_json_dict_(cls, pauli: pauli_gates.Pauli, qubit: 'cirq.Qid', **kwargs):
        return cls(pauli=pauli, qubit=qubit)