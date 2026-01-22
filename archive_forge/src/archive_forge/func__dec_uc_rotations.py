from __future__ import annotations
import math
import numpy as np
from qiskit.circuit.gate import Gate
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.quantumregister import QuantumRegister
from qiskit.exceptions import QiskitError
@staticmethod
def _dec_uc_rotations(angles, start_index, end_index, reversed_dec):
    """
        Calculates rotation angles for a uniformly controlled R_t gate with a C-NOT gate at
        the end of the circuit. The rotation angles of the gate R_t are stored in
        angles[start_index:end_index]. If reversed_dec == True, it decomposes the gate such that
        there is a C-NOT gate at the start of the circuit (in fact, the circuit topology for
        the reversed decomposition is the reversed one of the original decomposition)
        """
    interval_len_half = (end_index - start_index) // 2
    for i in range(start_index, start_index + interval_len_half):
        if not reversed_dec:
            angles[i], angles[i + interval_len_half] = UCPauliRotGate._update_angles(angles[i], angles[i + interval_len_half])
        else:
            angles[i + interval_len_half], angles[i] = UCPauliRotGate._update_angles(angles[i], angles[i + interval_len_half])
    if interval_len_half <= 1:
        return
    else:
        UCPauliRotGate._dec_uc_rotations(angles, start_index, start_index + interval_len_half, False)
        UCPauliRotGate._dec_uc_rotations(angles, start_index + interval_len_half, end_index, True)