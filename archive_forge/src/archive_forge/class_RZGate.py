from cmath import exp
from typing import Optional, Union
from qiskit.circuit.gate import Gate
from qiskit.circuit.controlledgate import ControlledGate
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.circuit.parameterexpression import ParameterValueType
class RZGate(Gate):
    """Single-qubit rotation about the Z axis.

    This is a diagonal gate. It can be implemented virtually in hardware
    via framechanges (i.e. at zero error and duration).

    Can be applied to a :class:`~qiskit.circuit.QuantumCircuit`
    with the :meth:`~qiskit.circuit.QuantumCircuit.rz` method.

    **Circuit symbol:**

    .. parsed-literal::

             ┌───────┐
        q_0: ┤ Rz(λ) ├
             └───────┘

    **Matrix Representation:**

    .. math::

        RZ(\\lambda) = \\exp\\left(-i\\frac{\\lambda}{2}Z\\right) =
            \\begin{pmatrix}
                e^{-i\\frac{\\lambda}{2}} & 0 \\\\
                0 & e^{i\\frac{\\lambda}{2}}
            \\end{pmatrix}

    .. seealso::

        :class:`~qiskit.circuit.library.standard_gates.U1Gate`
        This gate is equivalent to U1 up to a phase factor.

            .. math::

                U1(\\lambda) = e^{i{\\lambda}/2}RZ(\\lambda)

        Reference for virtual Z gate implementation:
        `1612.00858 <https://arxiv.org/abs/1612.00858>`_
    """

    def __init__(self, phi: ParameterValueType, label: Optional[str]=None, *, duration=None, unit='dt'):
        """Create new RZ gate."""
        super().__init__('rz', 1, [phi], label=label, duration=duration, unit=unit)

    def _define(self):
        """
        gate rz(phi) a { u1(phi) a; }
        """
        from qiskit.circuit.quantumcircuit import QuantumCircuit
        from .u1 import U1Gate
        q = QuantumRegister(1, 'q')
        theta = self.params[0]
        qc = QuantumCircuit(q, name=self.name, global_phase=-theta / 2)
        rules = [(U1Gate(theta), [q[0]], [])]
        for instr, qargs, cargs in rules:
            qc._append(instr, qargs, cargs)
        self.definition = qc

    def control(self, num_ctrl_qubits: int=1, label: Optional[str]=None, ctrl_state: Optional[Union[str, int]]=None, annotated: bool=False):
        """Return a (multi-)controlled-RZ gate.

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
            gate = CRZGate(self.params[0], label=label, ctrl_state=ctrl_state)
            gate.base_gate.label = self.label
        else:
            gate = super().control(num_ctrl_qubits=num_ctrl_qubits, label=label, ctrl_state=ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Return inverted RZ gate

        :math:`RZ(\\lambda)^{\\dagger} = RZ(-\\lambda)`

        Args:
            annotated: when set to ``True``, this is typically used to return an
                :class:`.AnnotatedOperation` with an inverse modifier set instead of a concrete
                :class:`.Gate`. However, for this class this argument is ignored as the inverse
                of this gate is always a :class:`.RZGate` with an inverted parameter value.

        Returns:
            RZGate: inverse gate.
        """
        return RZGate(-self.params[0])

    def __array__(self, dtype=None):
        """Return a numpy.array for the RZ gate."""
        import numpy as np
        ilam2 = 0.5j * float(self.params[0])
        return np.array([[exp(-ilam2), 0], [0, exp(ilam2)]], dtype=dtype)

    def power(self, exponent: float):
        """Raise gate to a power."""
        theta, = self.params
        return RZGate(exponent * theta)

    def __eq__(self, other):
        if isinstance(other, RZGate):
            return self._compare_parameters(other)
        return False