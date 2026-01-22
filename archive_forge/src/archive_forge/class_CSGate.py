from math import pi
from typing import Optional, Union
import numpy
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
@with_controlled_gate_array(_S_ARRAY, num_ctrl_qubits=1)
class CSGate(SingletonControlledGate):
    """Controlled-S gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cs` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──■──
             ┌─┴─┐
        q_1: ┤ S ├
             └───┘

    **Matrix representation:**

    .. math::

        CS \\ q_0, q_1 =
        I \\otimes |0 \\rangle\\langle 0| + S \\otimes |1 \\rangle\\langle 1|  =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 1 & 0 & 0 \\\\
                0 & 0 & 1 & 0 \\\\
                0 & 0 & 0 & i
            \\end{pmatrix}
    """

    def __init__(self, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CS gate."""
        super().__init__('cs', 2, [], label=label, num_ctrl_qubits=1, ctrl_state=ctrl_state, base_gate=SGate(label=_base_label), duration=duration, _base_label=_base_label, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key(num_ctrl_qubits=1)

    def _define(self):
        """
        gate cs a,b { h b; cp(pi/2) a,b; h b; }
        """
        from .p import CPhaseGate
        self.definition = CPhaseGate(theta=pi / 2).definition

    def inverse(self, annotated: bool=False):
        """Return inverse of CSGate (CSdgGate).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CSdgGate`.

        Returns:
            CSdgGate: inverse of :class:`.CSGate`
        """
        return CSdgGate(ctrl_state=self.ctrl_state)

    def power(self, exponent: float):
        """Raise gate to a power."""
        from .p import CPhaseGate
        return CPhaseGate(0.5 * numpy.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, CSGate) and self.ctrl_state == other.ctrl_state