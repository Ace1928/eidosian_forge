from __future__ import annotations
from typing import Optional, Union, Type
from math import ceil, pi
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int, with_gate_array, with_controlled_gate_array
class MCXRecursive(MCXGate):
    """Implement the multi-controlled X gate using recursion.

    Using a single ancilla qubit, the multi-controlled X gate is recursively split onto
    four sub-registers. This is done until we reach the 3- or 4-controlled X gate since
    for these we have a concrete implementation that do not require ancillas.
    """

    def __init__(self, num_ctrl_qubits: int, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        super().__init__(num_ctrl_qubits, label=label, ctrl_state=ctrl_state, _name='mcx_recursive', duration=duration, unit=unit, _base_label=None)

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str='recursion'):
        """Get the number of required ancilla qubits."""
        return MCXGate.get_num_ancilla_qubits(num_ctrl_qubits, mode)

    def inverse(self, annotated: bool=False):
        """Invert this gate. The MCX is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            MCXRecursive: inverse gate (self-inverse).
        """
        return MCXRecursive(num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state)

    def _define(self):
        """Define the MCX gate using recursion."""
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(self.num_qubits, name='q')
        qc = QuantumCircuit(q, name=self.name)
        if self.num_qubits == 4:
            qc._append(C3XGate(), q[:], [])
            self.definition = qc
        elif self.num_qubits == 5:
            qc._append(C4XGate(), q[:], [])
            self.definition = qc
        else:
            for instr, qargs, cargs in self._recurse(q[:-1], q_ancilla=q[-1]):
                qc._append(instr, qargs, cargs)
            self.definition = qc

    def _recurse(self, q, q_ancilla=None):
        if len(q) == 4:
            return [(C3XGate(), q[:], [])]
        if len(q) == 5:
            return [(C4XGate(), q[:], [])]
        if len(q) < 4:
            raise AttributeError('Something went wrong in the recursion, have less than 4 qubits.')
        num_ctrl_qubits = len(q) - 1
        middle = ceil(num_ctrl_qubits / 2)
        first_half = [*q[:middle], q_ancilla]
        second_half = [*q[middle:num_ctrl_qubits], q_ancilla, q[num_ctrl_qubits]]
        rule = []
        rule += self._recurse(first_half, q_ancilla=q[middle])
        rule += self._recurse(second_half, q_ancilla=q[middle - 1])
        rule += self._recurse(first_half, q_ancilla=q[middle])
        rule += self._recurse(second_half, q_ancilla=q[middle - 1])
        return rule