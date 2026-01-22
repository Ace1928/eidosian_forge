import math
from math import pi
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
class CRYGate(ControlledGate):
    """Controlled-RY gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cry` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ────■────
             ┌───┴───┐
        q_1: ┤ Ry(ϴ) ├
             └───────┘

    **Matrix representation:**

    .. math::

        \\newcommand{\\rotationangle}{\\frac{\\theta}{2}}

        CRY(\\theta)\\ q_0, q_1 =
            I \\otimes |0\\rangle\\langle 0| + RY(\\theta) \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0         & 0 & 0 \\\\
                0 & \\cos\\left(\\rotationangle\\right) & 0 & -\\sin\\left(\\rotationangle\\right) \\\\
                0 & 0         & 1 & 0 \\\\
                0 & \\sin\\left(\\rotationangle\\right) & 0 & \\cos\\left(\\rotationangle\\right)
            \\end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───────┐
            q_0: ┤ Ry(ϴ) ├
                 └───┬───┘
            q_1: ────■────

        .. math::

            \\newcommand{\\rotationangle}{\\frac{\\theta}{2}}

            CRY(\\theta)\\ q_1, q_0 =
            |0\\rangle\\langle 0| \\otimes I + |1\\rangle\\langle 1| \\otimes RY(\\theta) =
                \\begin{pmatrix}
                    1 & 0 & 0 & 0 \\\\
                    0 & 1 & 0 & 0 \\\\
                    0 & 0 & \\cos\\left(\\rotationangle\\right) & -\\sin\\left(\\rotationangle\\right) \\\\
                    0 & 0 & \\sin\\left(\\rotationangle\\right) & \\cos\\left(\\rotationangle\\right)
                \\end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CRY gate."""
        super().__init__('cry', 2, [theta], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=RYGate(theta, label=_base_label), duration=duration, unit=unit)

    def _define(self):
        """
        gate cry(lambda) a,b
        { u3(lambda/2,0,0) b; cx a,b;
          u3(-lambda/2,0,0) b; cx a,b;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(RYGate(self.params[0] / 2), [q[1]], []), (CXGate(), [q[0], q[1]], []), (RYGate(-self.params[0] / 2), [q[1]], []), (CXGate(), [q[0], q[1]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated: bool=False):
        """Return inverse CRY gate (i.e. with the negative rotation angle)

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CRYGate` with an inverted parameter value.

        Returns:
            CRYGate: inverse gate.
        ."""
        return CRYGate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        """Return a numpy.array for the CRY gate."""
        half_theta = float(self.params[0]) / 2
        cos = math.cos(half_theta)
        sin = math.sin(half_theta)
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0], [0, cos, 0, -sin], [0, 0, 1, 0], [0, sin, 0, cos]], dtype=dtype)
        else:
            return numpy.array([[cos, 0, -sin, 0], [0, 1, 0, 0], [sin, 0, cos, 0], [0, 0, 0, 1]], dtype=dtype)

    def __eq__(self, other):
        if isinstance(other, CRYGate):
            return self._compare_parameters(other) and self.ctrl_state == other.ctrl_state
        return False