import copy
import math
from cmath import exp
from typing import Optional, Union
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
class CUGate(ControlledGate):
    """Controlled-U gate (4-parameter two-qubit gate).

    This is a controlled version of the U gate (generic single qubit rotation),
    including a possible global phase :math:`e^{i\\gamma}` of the U gate.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cu` method.

    **Circuit symbol:**

    .. parsed-literal::

        q_0: ──────■──────
             ┌─────┴──────┐
        q_1: ┤ U(ϴ,φ,λ,γ) ├
             └────────────┘

    **Matrix representation:**

    .. math::

        \\newcommand{\\rotationangle}{\\frac{\\theta}{2}}

        CU(\\theta, \\phi, \\lambda, \\gamma)\\ q_0, q_1 =
            I \\otimes |0\\rangle\\langle 0| +
            e^{i\\gamma} U(\\theta,\\phi,\\lambda) \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & e^{i\\gamma}\\cos(\\rotationangle) &
                0 & -e^{i(\\gamma + \\lambda)}\\sin(\\rotationangle) \\\\
                0 & 0 & 1 & 0 \\\\
                0 & e^{i(\\gamma+\\phi)}\\sin(\\rotationangle) &
                0 & e^{i(\\gamma+\\phi+\\lambda)}\\cos(\\rotationangle)
            \\end{pmatrix}

    .. note::

        In Qiskit's convention, higher qubit indices are more significant
        (little endian convention). In many textbooks, controlled gates are
        presented with the assumption of more significant qubits as control,
        which in our case would be q_1. Thus a textbook matrix for this
        gate will be:

        .. parsed-literal::
                 ┌────────────┐
            q_0: ┤ U(ϴ,φ,λ,γ) ├
                 └─────┬──────┘
            q_1: ──────■───────

        .. math::

            \\newcommand{\\rotationangle}{\\frac{\\theta}{2}}
            CU(\\theta, \\phi, \\lambda, \\gamma)\\ q_1, q_0 =
            |0\\rangle\\langle 0| \\otimes I +
            e^{i\\gamma}|1\\rangle\\langle 1| \\otimes U(\\theta,\\phi,\\lambda) =
            \\begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & e^{i\\gamma} \\cos(\\rotationangle) & -e^{i(\\gamma + \\lambda)}\\sin(\\rotationangle) \\\\
            0 & 0 &
            e^{i(\\gamma + \\phi)}\\sin(\\rotationangle) & e^{i(\\gamma + \\phi+\\lambda)}\\cos(\\rotationangle)
            \\end{pmatrix}
    """

    def __init__(self, theta: ParameterValueType, phi: ParameterValueType, lam: ParameterValueType, gamma: ParameterValueType, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CU gate."""
        super().__init__('cu', 2, [theta, phi, lam, gamma], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=UGate(theta, phi, lam, label=_base_label), duration=duration, unit=unit)

    def _define(self):
        """
        gate cu(theta,phi,lambda,gamma) c, t
        { phase(gamma) c;
          phase((lambda+phi)/2) c;
          phase((lambda-phi)/2) t;
          cx c,t;
          u(-theta/2,0,-(phi+lambda)/2) t;
          cx c,t;
          u(theta/2,phi,0) t;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        qc.p(self.params[3], 0)
        qc.p((self.params[2] + self.params[1]) / 2, 0)
        qc.p((self.params[2] - self.params[1]) / 2, 1)
        qc.cx(0, 1)
        qc.u(-self.params[0] / 2, 0, -(self.params[1] + self.params[2]) / 2, 1)
        qc.cx(0, 1)
        qc.u(self.params[0] / 2, self.params[1], 0, 1)
        self.definition = qc

    def inverse(self, annotated: bool=False):
        """Return inverted CU gate.

        :math:`CU(\\theta,\\phi,\\lambda,\\gamma)^{\\dagger} = CU(-\\theta,-\\phi,-\\lambda,-\\gamma))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CUGate` with inverse parameter
                values.

        Returns:
            CUGate: inverse gate.
        """
        return CUGate(-self.params[0], -self.params[2], -self.params[1], -self.params[3], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        """Return a numpy.array for the CU gate."""
        theta, phi, lam, gamma = (float(param) for param in self.params)
        cos = numpy.cos(theta / 2)
        sin = numpy.sin(theta / 2)
        a = numpy.exp(1j * gamma) * cos
        b = -numpy.exp(1j * (gamma + lam)) * sin
        c = numpy.exp(1j * (gamma + phi)) * sin
        d = numpy.exp(1j * (gamma + phi + lam)) * cos
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0], [0, a, 0, b], [0, 0, 1, 0], [0, c, 0, d]], dtype=dtype)
        else:
            return numpy.array([[a, 0, b, 0], [0, 1, 0, 0], [c, 0, d, 0], [0, 0, 0, 1]], dtype=dtype)

    @property
    def params(self):
        return _CUGateParams(self)

    @params.setter
    def params(self, parameters):
        super(ControlledGate, type(self)).params.fset(self, parameters)
        self.base_gate.params = parameters[:-1]

    def __deepcopy__(self, memo=None):
        memo = memo if memo is not None else {}
        out = super().__deepcopy__(memo)
        out._params = copy.deepcopy(out._params, memo)
        return out