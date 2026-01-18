from __future__ import annotations
from qiskit.circuit.bit import Bit
from qiskit.circuit import QuantumCircuit, QuantumRegister, AncillaRegister
from .adder import Adder

        Args:
            num_state_qubits: The size of the register.
            kind: The kind of adder, can be ``'full'`` for a full adder, ``'half'`` for a half
                adder, or ``'fixed'`` for a fixed-sized adder. A full adder includes both carry-in
                and carry-out, a half only carry-out, and a fixed-sized adder neither carry-in
                nor carry-out.
            name: The name of the circuit.

        Raises:
            ValueError: If ``num_state_qubits`` is lower than 1.
        