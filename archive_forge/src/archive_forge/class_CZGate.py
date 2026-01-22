from math import pi
from typing import Optional, Union
import numpy
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from .p import PhaseGate
@with_controlled_gate_array(_Z_ARRAY, num_ctrl_qubits=1)
class CZGate(SingletonControlledGate):
    """Controlled-Z gate.

    This is a Clifford and symmetric gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cz` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ─■─
              │
        q_1: ─■─

    **Matrix representation:**

    .. math::

        CZ\\ q_0, q_1 =
            I \\otimes |0\\rangle\\langle 0| + Z \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 1 & 0 & 0 \\\\
                0 & 0 & 1 & 0 \\\\
                0 & 0 & 0 & -1
            \\end{pmatrix}

    In the computational basis, this gate flips the phase of
    the target qubit if the control qubit is in the :math:`|1\\rangle` state.
    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CZ gate."""
        super().__init__('cz', 2, [], label=label, num_ctrl_qubits=1, ctrl_state=ctrl_state, base_gate=ZGate(label=_base_label), duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """
        gate cz a,b { h b; cx a,b; h b; }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .h import HGate
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(HGate(), [q[1]], []), (CXGate(), [q[0], q[1]], []), (HGate(), [q[1]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated: bool=False):
        """Return inverted CZ gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            CZGate: inverse gate (self-inverse).
        """
        return CZGate(ctrl_state=self.ctrl_state)

    def __eq__(self, other):
        return isinstance(other, CZGate) and self.ctrl_state == other.ctrl_state