from __future__ import annotations
from typing import Optional, Union, Type
from math import ceil, pi
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int, with_gate_array, with_controlled_gate_array
class MCXGate(ControlledGate):
    """The general, multi-controlled X gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.mcx` method.
    """

    def __new__(cls, num_ctrl_qubits: Optional[int]=None, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create a new MCX instance.

        Depending on the number of controls and which mode of the MCX, this creates an
        explicit CX, CCX, C3X or C4X instance or a generic MCX gate.
        """
        explicit: dict[int, Type[ControlledGate]] = {1: CXGate, 2: CCXGate}
        gate_class = explicit.get(num_ctrl_qubits, None)
        if gate_class is not None:
            gate = gate_class.__new__(gate_class, label=label, ctrl_state=ctrl_state, _base_label=_base_label)
            gate.__init__(label=label, ctrl_state=ctrl_state, _base_label=_base_label, duration=duration, unit=unit)
            return gate
        return super().__new__(cls)

    def __init__(self, num_ctrl_qubits: int, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _name='mcx', _base_label=None):
        """Create new MCX gate."""
        num_ancilla_qubits = self.__class__.get_num_ancilla_qubits(num_ctrl_qubits)
        super().__init__(_name, num_ctrl_qubits + 1 + num_ancilla_qubits, [], num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, base_gate=XGate(label=_base_label))

    def inverse(self, annotated: bool=False):
        """Invert this gate. The MCX is its own inverse.

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            MCXGate: inverse gate (self-inverse).
        """
        return MCXGate(num_ctrl_qubits=self.num_ctrl_qubits, ctrl_state=self.ctrl_state)

    @staticmethod
    def get_num_ancilla_qubits(num_ctrl_qubits: int, mode: str='noancilla') -> int:
        """Get the number of required ancilla qubits without instantiating the class.

        This staticmethod might be necessary to check the number of ancillas before
        creating the gate, or to use the number of ancillas in the initialization.
        """
        if mode == 'noancilla':
            return 0
        if mode in ['recursion', 'advanced']:
            return int(num_ctrl_qubits > 4)
        if mode[:7] == 'v-chain' or mode[:5] == 'basic':
            return max(0, num_ctrl_qubits - 2)
        raise AttributeError(f'Unsupported mode ({mode}) specified!')

    def _define(self):
        """The standard definition used the Gray code implementation."""
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(self.num_qubits, name='q')
        qc = QuantumCircuit(q)
        qc._append(MCXGrayCode(self.num_ctrl_qubits), q[:], [])
        self.definition = qc

    @property
    def num_ancilla_qubits(self):
        """The number of ancilla qubits."""
        return self.__class__.get_num_ancilla_qubits(self.num_ctrl_qubits)

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, annotated: bool=False):
        """Return a multi-controlled-X gate with more control lines.

        Args:
            num_ctrl_qubits: number of control qubits.
            label: An optional label for the gate [Default: ``None``]
            ctrl_state: control state expressed as integer,
                string (e.g.``'110'``), or ``None``. If ``None``, use all 1s.
            annotated: indicates whether the controlled gate can be implemented
                as an annotated gate.

        Returns:
            ControlledGate: controlled version of this gate.
        """
        if not annotated and ctrl_state is None:
            gate = self.__class__(self.num_ctrl_qubits + num_ctrl_qubits, label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(num_ctrl_qubits, label=label, ctrl_state=ctrl_state)
        return gate