from copy import deepcopy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import calc_inverse_matrix
from qiskit.synthesis.linear.linear_depth_lnn import _optimize_cx_circ_depth_5n_line
def _swap_plus(instructions, seq):
    """
    Given CX instructions (c.f. Thm 7.1, [1]) and the labels of all boxes,
    Return a list of labels of the boxes that is SWAP+ in descending order
        * Assumes the instruction gives gates in the order from top to bottom,
          from left to right
        * SWAP+ is defined in section 3.A. of [2]. Note the northwest
          diagonalization procedure of [1] consists exactly n layers of boxes,
          each being either a SWAP or a SWAP+. That is, each northwest
          diagonalization circuit can be uniquely represented by which of its
          n(n-1)/2 boxes are SWAP+ and which are SWAP.
    """
    instr = deepcopy(instructions)
    swap_plus = set()
    for i, j in reversed(seq):
        cnot_1 = instr.pop()
        instr.pop()
        if instr == [] or instr[-1] != cnot_1:
            swap_plus.add((i, j))
        else:
            instr.pop()
    return swap_plus