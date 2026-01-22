from copy import deepcopy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import calc_inverse_matrix
from qiskit.synthesis.linear.linear_depth_lnn import _optimize_cx_circ_depth_5n_line
def _apply_phase_to_nw_circuit(n, phase_schedule, seq, swap_plus):
    """
    Given
        Width of the circuit (int n)
        A CZ circuit, represented by the n*n phase schedule phase_schedule
        A CX circuit, represented by box-labels (seq) and whether the box is SWAP+ (swap_plus)
            *   This circuit corresponds to the CX tranformation that tranforms a matrix to
                a NW matrix (c.f. Prop.7.4, [1])
            *   SWAP+ is defined in section 3.A. of [2].
            *   As previously noted, the northwest diagonalization procedure of [1] consists
                of exactly n layers of boxes, each being either a SWAP or a SWAP+. That is,
                each northwest diagonalization circuit can be uniquely represented by which
                of its n(n-1)/2 boxes are SWAP+ and which are SWAP.
    Return a QuantumCircuit that computes the phase scheudle S inside CX
    """
    cir = QuantumCircuit(n)
    wires = list(zip(range(n), range(1, n)))
    wires = wires[::2] + wires[1::2]
    for i, (j, k) in zip(range(len(seq) - 1, -1, -1), reversed(seq)):
        w1, w2 = wires[i % (n - 1)]
        p = phase_schedule[j, k]
        if (j, k) not in swap_plus:
            cir.cx(w1, w2)
        cir.cx(w2, w1)
        if p % 4 == 0:
            pass
        elif p % 4 == 1:
            cir.sdg(w2)
        elif p % 4 == 2:
            cir.z(w2)
        else:
            cir.s(w2)
        cir.cx(w1, w2)
    for i in range(n):
        p = phase_schedule[n - 1 - i, n - 1 - i]
        if p % 4 == 0:
            continue
        if p % 4 == 1:
            cir.sdg(i)
        elif p % 4 == 2:
            cir.z(i)
        else:
            cir.s(i)
    return cir