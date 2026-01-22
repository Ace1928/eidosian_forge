from typing import Dict, Optional, Sequence
import numpy as np
import cirq
from cirq import circuits
def _sparse_computational_basis_map(inputs: Sequence[int], circuit: circuits.Circuit) -> Dict[int, int]:
    amps = [np.exp(1j * i / len(inputs)) / len(inputs) ** 0.5 for i in range(len(inputs))]
    input_state = np.zeros(1 << len(circuit.all_qubits()), dtype=np.complex128)
    for k, amp in zip(inputs, amps):
        input_state[k] = amp
    output_state = cirq.final_state_vector(circuit, initial_state=input_state)
    actual_map = {}
    for k, amp in zip(inputs, amps):
        for i, amp2 in enumerate(output_state):
            if abs(amp2 - amp) < 1e-05:
                actual_map[k] = i
    return actual_map