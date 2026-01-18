from __future__ import annotations
from collections.abc import Callable
from qiskit import circuit
from qiskit.circuit import ControlledGate, Gate, QuantumRegister, QuantumCircuit
from qiskit.exceptions import QiskitError
from ..standard_gates import XGate, YGate, ZGate, HGate, TGate, TdgGate, SGate, SdgGate
Get the rule for the CCX V-chain.

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
        