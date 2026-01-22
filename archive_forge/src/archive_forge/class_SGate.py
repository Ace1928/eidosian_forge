from math import pi
from typing import Optional, Union
import numpy
from qiskit.circuit.singleton import SingletonGate, SingletonControlledGate, stdlib_singleton_key
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import with_gate_array, with_controlled_gate_array
@with_gate_array(_S_ARRAY)
class SGate(SingletonGate):
    """Single qubit S gate (Z**0.5).

    It induces a :math:`\\pi/2` phase, and is sometimes called the P gate (phase).

    This is a Clifford gate and a square-root of Pauli-Z.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.s` method.

    **Matrix Representation:**

    .. math::

        S = \\begin{pmatrix}
                1 & 0 \\\\
                0 & i
            \\end{pmatrix}

    **Circuit symbol:**

    .. parsed-literal::

             ┌───┐
        q_0: ┤ S ├
             └───┘

    Equivalent to a :math:`\\pi/2` radian rotation about the Z axis.
    """

    def __init__(self, label: Optional[str]=None, *, duration=None, unit='dt'):
        """Create new S gate."""
        super().__init__('s', 1, [], label=label, duration=duration, unit=unit)
    _singleton_lookup_key = stdlib_singleton_key()

    def _define(self):
        """
        gate s a { u1(pi/2) a; }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        q = QuantumRegister(1, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(pi / 2), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated: bool=False):
        """Return inverse of S (SdgGate).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.SdgGate`.

        Returns:
            SdgGate: inverse of :class:`.SGate`
        """
        return SdgGate()

    def power(self, exponent: float):
        """Raise gate to a power."""
        from .p import PhaseGate
        return PhaseGate(0.5 * numpy.pi * exponent)

    def __eq__(self, other):
        return isinstance(other, SGate)