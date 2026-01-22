from math import pi
from typing import Optional, Union
import numpy
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from .p import PhaseGate
@with_controlled_gate_array(_Z_ARRAY, num_ctrl_qubits=2, cached_states=(3,))
class CCZGate(SingletonControlledGate):
    """CCZ gate.

    This is a symmetric gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.ccz` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ─■─
              │
        q_1: ─■─
              │
        q_2: ─■─

    **Matrix representation:**

    .. math::

        CCZ\\ q_0, q_1, q_2 =
            I \\otimes I \\otimes |0\\rangle\\langle 0| + CZ \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 0 & 1 & 0 \\\\
                0 & 0 & 0 & 0 & 0 & 0 & 0 & -1
            \\end{pmatrix}

    In the computational basis, this gate flips the phase of
    the target qubit if the control qubits are in the :math:`|11\\rangle` state.
    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CCZ gate."""
        super().__init__('ccz', 3, [], label=label, num_ctrl_qubits=2, ctrl_state=ctrl_state, base_gate=ZGate(label=_base_label), duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=2)

    def _define(self):
        """
        gate ccz a,b,c { h c; ccx a,b,c; h c; }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .x import CCXGate
        q = QuantumRegister(3, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(HGate(), [q[2]], []), (CCXGate(), [q[0], q[1], q[2]], []), (HGate(), [q[2]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated: bool=False):
        """Return inverted CCZ gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CCZGate: inverse gate (self-inverse).
        """
        return CCZGate(ctrl_state=self.ctrl_state)

    def __eq__(self, other):
        return isinstance(other, CCZGate) and self.ctrl_state == other.ctrl_state