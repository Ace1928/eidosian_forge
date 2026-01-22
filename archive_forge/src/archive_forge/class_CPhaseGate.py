from __future__ import annotations
from cmath import exp
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
class CPhaseGate(ControlledGate):
    """Controlled-Phase gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the control state.

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.cp` method.

    **Circuit symbol:**

    .. parsed-literal::


        q_0: ─■──
              │λ
        q_1: ─■──


    **Matrix representation:**

    .. math::

        CPhase =
            I \\otimes |0\\rangle\\langle 0| + P \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 1 & 0 & 0 \\\\
                0 & 0 & 1 & 0 \\\\
                0 & 0 & 0 & e^{i\\lambda}
            \\end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CRZGate`:
        Due to the global phase difference in the matrix definitions
        of Phase and RZ, CPhase and CRZ are different gates with a relative
        phase difference.
    """

    def __init__(self, theta: ParameterValueType, label: str | None=None, ctrl_state: str | int | None=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CPhase gate."""
        super().__init__('cp', 2, [theta], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=PhaseGate(theta, label=_base_label), duration=duration, unit=unit)

    def _define(self):
        """
        gate cphase(lambda) a,b
        { phase(lambda/2) a; cx a,b;
          phase(-lambda/2) b; cx a,b;
          phase(lambda/2) b;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        qc.p(self.params[0] / 2, 0)
        qc.cx(0, 1)
        qc.p(-self.params[0] / 2, 1)
        qc.cx(0, 1)
        qc.p(self.params[0] / 2, 1)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: str | None=None, ctrl_state: str | int | None=None, annotated: bool=False):
        """Controlled version of this gate.

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
            gate = MCPhaseGate(self.params[0], num_ctrl_qubits=num_ctrl_qubits + 1, label=label)
            gate.base_gate.label = self.label
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Return inverted CPhase gate (:math:`CPhase(\\lambda)^{\\dagger} = CPhase(-\\lambda)`)"""
        return CPhaseGate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        """Return a numpy.array for the CPhase gate."""
        eith = exp(1j * float(self.params[0]))
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, eith]], dtype=dtype)
        return numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, eith, 0], [0, 0, 0, 1]], dtype=dtype)

    def power(self, exponent: float):
        """Raise gate to a power."""
        theta, = self.params
        return CPhaseGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, CPhaseGate):
            return self._compare_parameters(other) and self.ctrl_state == other.ctrl_state
        return False