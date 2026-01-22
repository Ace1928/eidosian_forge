from math import sqrt, pi
from typing import Optional, Union
import numpy
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
@with_gate_array(_H_ARRAY)
class HGate(SingletonGate):
    """Single-qubit Hadamard gate.

    This gate is a \\pi rotation about the X+Z axis, and has the effect of
    changing computation basis from :math:`|0\\rangle,|1\\rangle` to
    :math:`|+\\rangle,|-\\rangle` and vice-versa.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.h` method.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ H ├
             └───┘

    **Matrix Representation:**

    .. math::

        H = \\frac{1}{\\sqrt{2}}
            \\begin{pmatrix}
                1 & 1 \\\\
                1 & -1
            \\end{pmatrix}
    """

    def __init__(self, label: Optional[str]=None, *, duration=None, unit='dt'):
        """Create new H gate."""
        super().__init__('h', 1, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate h a { u2(0,pi) a; }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u2 import U2Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U2Gate(0, pi), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[int, str]]=None, annotated: bool=False):
        """Return a (multi-)controlled-H gate.

        One control qubit returns a CH gate.

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
        if not annotated and num_ctrl_qubits == 1:
            gate = CHGate(label=label, ctrl_state=ctrl_state, _base_label=self.label)
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Return inverted H gate (itself).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as this gate
                is self-inverse.

        Returns:
            HGate: inverse gate (self-inverse).
        """
        return HGate()

    def __eq__(self, other):
        return isinstance(other, HGate)