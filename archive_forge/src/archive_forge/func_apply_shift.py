from __future__ import annotations
import numpy as np
from qiskit.circuit.quantumcircuit import QuantumCircuit
from qiskit.circuit.library.standard_gates import RXGate, RYGate, RZGate
def apply_shift(shift_name, coordinate):
    """
    Given a shift type and a canonical coordinate, applies the shift and
    describes a circuit which enacts the shift + a global phase shift.
    """
    shift_scalars, shift_phase_shift, source_shift_gates = shift_options[shift_name]
    shifted_coord = [np.pi / 2 * x + y for x, y in zip(shift_scalars, coordinate)]
    source_shift = QuantumCircuit(2)
    for gate in source_shift_gates:
        source_shift.append(gate(np.pi), [0])
        source_shift.append(gate(np.pi), [1])
    return (shifted_coord, source_shift, shift_phase_shift)