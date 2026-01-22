import math
from math import pi
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
class CRXGate(ControlledGate):
    """Controlled-RX gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.crx` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ────■────
             ┌───┴───┐
        q_1: ┤ Rx(ϴ) ├
             └───────┘

    **Matrix representation:**

    .. math::

        \\newcommand{\\rotationangle}{\\frac{\\theta}{2}}

        CRX(\\theta)\\ q_0, q_1 =
            I \\otimes |0\\rangle\\langle 0| + RX(\\theta) \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & \\cos\\left(\\rotationangle\\right) & 0 & -i\\sin\\left(\\rotationangle\\right) \\\\
                0 & 0 & 1 & 0 \\\\
                0 & -i\\sin\\left(\\rotationangle\\right) & 0 & \\cos\\left(\\rotationangle\\right)
            \\end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌───────┐
            q_0: ┤ Rx(ϴ) ├
                 └───┬───┘
            q_1: ────■────

        .. math::

            \\newcommand{\\rotationangle}{\\frac{\\theta}{2}}

            CRX(\\theta)\\ q_1, q_0 =
            |0\\rangle\\langle0| \\otimes I + |1\\rangle\\langle1| \\otimes RX(\\theta) =
                \\begin{pmatrix}
                    1 & 0 & 0 & 0 \\\\
                    0 & 1 & 0 & 0 \\\\
                    0 & 0 & \\cos\\left(\\rotationangle\\right)   & -i\\sin\\left(\\rotationangle\\right) \\\\
                    0 & 0 & -i\\sin\\left(\\rotationangle\\right) & \\cos\\left(\\rotationangle\\right)
                \\end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CRX gate."""
        super().__init__('crx', 2, [theta], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=RXGate(theta, label=_base_label))

    def _define(self):
        """
        gate cu3(theta,phi,lambda) c, t
        { u1(pi/2) t;
          cx c,t;
          u3(-theta/2,0,0) t;
          cx c,t;
          u3(theta/2,-pi/2,0) t;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        from .u3 import U3Gate
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(pi / 2), [q[1]], []), (CXGate(), [q[0], q[1]], []), (U3Gate(-self.params[0] / 2, 0, 0), [q[1]], []), (CXGate(), [q[0], q[1]], []), (U3Gate(self.params[0] / 2, -pi / 2, 0), [q[1]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def inverse(self, annotated: bool=False):
        """Return inverse CRX gate (i.e. with the negative rotation angle).

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CRXGate` with an inverted parameter value.

        Returns:
            CRXGate: inverse gate.
        """
        return CRXGate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        """Return a numpy.array for the CRX gate."""
        half_theta = float(self.params[0]) / 2
        cos = math.cos(half_theta)
        isin = 1j * math.sin(half_theta)
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0], [0, cos, 0, -isin], [0, 0, 1, 0], [0, -isin, 0, cos]], dtype=dtype)
        else:
            return numpy.array([[cos, 0, -isin, 0], [0, 1, 0, 0], [-isin, 0, cos, 0], [0, 0, 0, 1]], dtype=dtype)

    def __eq__(self, other):
        if isinstance(other, CRXGate):
            return self._compare_parameters(other) and self.ctrl_state == other.ctrl_state
        return False