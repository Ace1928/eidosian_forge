from __future__ import annotations
from collections.abc import Callable
from qiskit import circuit
from qiskit.circuit import ControlledGate, Gate, QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from ..standard_gates import XGate, YGate, ZGate, HGate, TGate, TdgGate, SGate, SdgGate
class MCMTVChain(MCMT):
    """The MCMT implementation using the CCX V-chain.

    This implementation requires ancillas but is decomposed into a much shallower circuit
    than the default implementation in :class:`~qiskit.circuit.library.MCMT`.

    **Expanded Circuit:**

    .. plot::

       from qiskit.circuit.library import MCMTVChain, ZGate
       from qiskit.visualization.library import _generate_circuit_library_visualization
       circuit = MCMTVChain(ZGate(), 2, 2)
       _generate_circuit_library_visualization(circuit.decompose())

    **Examples:**

        >>> from qiskit.circuit.library import HGate
        >>> MCMTVChain(HGate(), 3, 2).draw()

        q_0: ──■────────────────────────■──
               │                        │
        q_1: ──■────────────────────────■──
               │                        │
        q_2: ──┼────■──────────────■────┼──
               │    │  ┌───┐       │    │
        q_3: ──┼────┼──┤ H ├───────┼────┼──
               │    │  └─┬─┘┌───┐  │    │
        q_4: ──┼────┼────┼──┤ H ├──┼────┼──
             ┌─┴─┐  │    │  └─┬─┘  │  ┌─┴─┐
        q_5: ┤ X ├──■────┼────┼────■──┤ X ├
             └───┘┌─┴─┐  │    │  ┌─┴─┐└───┘
        q_6: ─────┤ X ├──■────■──┤ X ├─────
                  └───┘          └───┘
    """

    def _build(self):
        """Define the MCMT gate."""
        control_qubits = self.qubits[:self.num_ctrl_qubits]
        target_qubits = self.qubits[self.num_ctrl_qubits:self.num_ctrl_qubits + self.num_target_qubits]
        ancilla_qubits = self.qubits[self.num_ctrl_qubits + self.num_target_qubits:]
        if len(ancilla_qubits) > 0:
            master_control = ancilla_qubits[-1]
        else:
            master_control = control_qubits[0]
        self._ccx_v_chain_rule(control_qubits, ancilla_qubits, reverse=False)
        for qubit in target_qubits:
            self.append(self.gate.control(), [master_control, qubit], [])
        self._ccx_v_chain_rule(control_qubits, ancilla_qubits, reverse=True)

    @property
    def num_ancilla_qubits(self):
        """Return the number of ancilla qubits required."""
        return max(0, self.num_ctrl_qubits - 1)

    def _ccx_v_chain_rule(self, control_qubits: QuantumRegister | list[circuit.Qubit], ancilla_qubits: QuantumRegister | list[circuit.Qubit], reverse: bool=False) -> None:
        """Get the rule for the CCX V-chain.

        The CCX V-chain progressively computes the CCX of the control qubits and puts the final
        result in the last ancillary qubit.

        Args:
            control_qubits: The control qubits.
            ancilla_qubits: The ancilla qubits.
            reverse: If True, compute the chain down to the qubit. If False, compute upwards.

        Returns:
            The rule for the (reversed) CCX V-chain.

        Raises:
            QiskitError: If an insufficient number of ancilla qubits was provided.
        """
        if len(ancilla_qubits) == 0:
            return
        if len(ancilla_qubits) < len(control_qubits) - 1:
            raise QiskitError('Insufficient number of ancilla qubits.')
        iterations = list(enumerate(range(2, len(control_qubits))))
        if not reverse:
            self.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0])
            for i, j in iterations:
                self.ccx(control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1])
        else:
            for i, j in reversed(iterations):
                self.ccx(control_qubits[j], ancilla_qubits[i], ancilla_qubits[i + 1])
            self.ccx(control_qubits[0], control_qubits[1], ancilla_qubits[0])

    def inverse(self, annotated: bool=False):
        return MCMTVChain(self.gate, self.num_ctrl_qubits, self.num_target_qubits)