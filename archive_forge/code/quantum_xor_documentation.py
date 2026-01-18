from typing import Optional
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.circuit.exceptions import CircuitError
Return a circuit implementing bitwise xor.

        Args:
            num_qubits: the width of circuit.
            amount: the xor amount in decimal form.
            seed: random seed in case a random xor is requested.

        Raises:
            CircuitError: if the xor bitstring exceeds available qubits.

        Reference Circuit:
            .. plot::

               from qiskit.circuit.library import XOR
               from qiskit.visualization.library import _generate_circuit_library_visualization
               circuit = XOR(5, seed=42)
               _generate_circuit_library_visualization(circuit)
        