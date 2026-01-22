from __future__ import annotations
from collections.abc import Callable
from qiskit import circuit
from qiskit.circuit import ControlledGate, Gate, QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from ..standard_gates import XGate, YGate, ZGate, HGate, TGate, TdgGate, SGate, SdgGate
class MCMT(QuantumCircuit):
    """The multi-controlled multi-target gate, for an arbitrary singly controlled target gate.

    For example, the H gate controlled on 3 qubits and acting on 2 target qubit is represented as:

    .. parsed-literal::

        ───■────
           │
        ───■────
           │
        ───■────
        ┌──┴───┐
        ┤0     ├
        │  2-H │
        ┤1     ├
        └──────┘

    This default implementations requires no ancilla qubits, by broadcasting the target gate
    to the number of target qubits and using Qiskit's generic control routine to control the
    broadcasted target on the control qubits. If ancilla qubits are available, a more efficient
    variant using the so-called V-chain decomposition can be used. This is implemented in
    :class:`~qiskit.circuit.library.MCMTVChain`.
    """

    def __init__(self, gate: Gate | Callable[[QuantumCircuit, circuit.Qubit, circuit.Qubit], circuit.Instruction], num_ctrl_qubits: int, num_target_qubits: int) -> None:
        """Create a new multi-control multi-target gate.

        Args:
            gate: The gate to be applied controlled on the control qubits and applied to the target
                qubits. Can be either a Gate or a circuit method.
                If it is a callable, it will be casted to a Gate.
            num_ctrl_qubits: The number of control qubits.
            num_target_qubits: The number of target qubits.

        Raises:
            AttributeError: If the gate cannot be casted to a controlled gate.
            AttributeError: If the number of controls or targets is 0.
        """
        if num_ctrl_qubits == 0 or num_target_qubits == 0:
            raise AttributeError('Need at least one control and one target qubit.')
        self.gate = self._identify_gate(gate)
        self.num_ctrl_qubits = num_ctrl_qubits
        self.num_target_qubits = num_target_qubits
        num_qubits = num_ctrl_qubits + num_target_qubits + self.num_ancilla_qubits
        super().__init__(num_qubits, name='mcmt')
        self._label = f'{num_target_qubits}-{self.gate.name.capitalize()}'
        self._build()

    def _build(self):
        """Define the MCMT gate without ancillas."""
        if self.num_target_qubits == 1:
            broadcasted_gate = self.gate
        else:
            broadcasted = QuantumCircuit(self.num_target_qubits, name=self._label)
            for target in list(range(self.num_target_qubits)):
                broadcasted.append(self.gate, [target], [])
            broadcasted_gate = broadcasted.to_gate()
        mcmt_gate = broadcasted_gate.control(self.num_ctrl_qubits)
        self.append(mcmt_gate, self.qubits, [])

    @property
    def num_ancilla_qubits(self):
        """Return the number of ancillas."""
        return 0

    def _identify_gate(self, gate):
        """Case the gate input to a gate."""
        valid_gates = {'ch': HGate(), 'cx': XGate(), 'cy': YGate(), 'cz': ZGate(), 'h': HGate(), 's': SGate(), 'sdg': SdgGate(), 'x': XGate(), 'y': YGate(), 'z': ZGate(), 't': TGate(), 'tdg': TdgGate()}
        if isinstance(gate, ControlledGate):
            base_gate = gate.base_gate
        elif isinstance(gate, Gate):
            if gate.num_qubits != 1:
                raise AttributeError('Base gate must act on one qubit only.')
            base_gate = gate
        elif isinstance(gate, QuantumCircuit):
            if gate.num_qubits != 1:
                raise AttributeError('The circuit you specified as control gate can only have one qubit!')
            base_gate = gate.to_gate()
        else:
            if callable(gate):
                name = gate.__name__
            elif isinstance(gate, str):
                name = gate
            else:
                raise AttributeError(f'Invalid gate specified: {gate}')
            base_gate = valid_gates[name]
        return base_gate

    def control(self, num_ctrl_qubits=1, label=None, ctrl_state=None, annotated=False):
        """Return the controlled version of the MCMT circuit."""
        if not annotated and ctrl_state is None:
            gate = MCMT(self.gate, self.num_ctrl_qubits + num_ctrl_qubits, self.num_target_qubits)
        else:
            gate = super().control(num_ctrl_qubits, label, ctrl_state, annotated=annotated)
        return gate

    def inverse(self, annotated: bool=False):
        """Return the inverse MCMT circuit, which is itself."""
        return MCMT(self.gate, self.num_ctrl_qubits, self.num_target_qubits)