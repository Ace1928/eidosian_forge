from __future__ import annotations
from cmath import exp
import numpy
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.gate import Gate
from qiskit.circuit.parameterexpression import ParameterValueType
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit._utils import _ctrl_state_to_int
class CU1Gate(ControlledGate):
    """Controlled-U1 gate.

    This is a diagonal and symmetric gate that induces a
    phase on the state of the target qubit, depending on the control state.

    **Circuit symbol:**

    .. parsed-literal::


        q_0: ─■──
              │λ
        q_1: ─■──


    **Matrix representation:**

    .. math::

        CU1(\\lambda) =
            I \\otimes |0\\rangle\\langle 0| + U1 \\otimes |1\\rangle\\langle 1| =
            \\begin{pmatrix}
                1 & 0 & 0 & 0 \\\\
                0 & 1 & 0 & 0 \\\\
                0 & 0 & 1 & 0 \\\\
                0 & 0 & 0 & e^{i\\lambda}
            \\end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.CRZGate`:
        Due to the global phase difference in the matrix definitions
        of U1 and RZ, CU1 and CRZ are different gates with a relative
        phase difference.
    """

    def __init__(self, theta: ParameterValueType, label: str | None=None, ctrl_state: str | int | None=None, *, duration=None, unit='dt', _base_label=None):
        """Create new CU1 gate."""
        super().__init__('cu1', 2, [theta], num_ctrl_qubits=1, label=label, ctrl_state=ctrl_state, base_gate=U1Gate(theta, label=_base_label), duration=duration, unit=unit)

    def _define(self):
        """
        gate cu1(lambda) a,b
        { u1(lambda/2) a; cx a,b;
          u1(-lambda/2) b; cx a,b;
          u1(lambda/2) b;
        }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .x import CXGate
        q = QuantumRegister(2, 'q')
        qc = QuantumCircuit(q, name=self.name)
        rules = [(U1Gate(self.params[0] / 2), [q[0]], []), (CXGate(), [q[0], q[1]], []), (U1Gate(-self.params[0] / 2), [q[1]], []), (CXGate(), [q[0], q[1]], []), (U1Gate(self.params[0] / 2), [q[1]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
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
            gate = MCU1Gate(self.params[0], num_ctrl_qubits=num_ctrl_qubits + 1, label=label)
            gate.base_gate.label = self.label
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Return inverted CU1 gate (:math:`CU1(\\lambda)^{\\dagger} = CU1(-\\lambda))`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.CU1Gate` with inverse parameter
                values.

        Returns:
            CU1Gate: inverse gate.
        """
        return CU1Gate(-self.params[0], ctrl_state=self.ctrl_state)

    def __array__(self, dtype=None):
        """Return a numpy.array for the CU1 gate."""
        eith = exp(1j * float(self.params[0]))
        if self.ctrl_state:
            return numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, eith]], dtype=dtype)
        else:
            return numpy.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, eith, 0], [0, 0, 0, 1]], dtype=dtype)