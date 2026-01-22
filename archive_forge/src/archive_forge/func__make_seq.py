from copy import deepcopy
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import calc_inverse_matrix
from qiskit.synthesis.linear.linear_depth_lnn import _optimize_cx_circ_depth_5n_line
def _make_seq(n):
    """
    Given the width of the circuit n,
    Return the labels of the boxes in order from left to right, top to bottom
    (c.f. Fig.2, [2])
    """
    seq = []
    wire_labels = list(range(n - 1, -1, -1))
    for i in range(n):
        wire_labels_new = _shuffle(wire_labels, n % 2) if i % 2 == 0 else wire_labels[0:1] + _shuffle(wire_labels[1:], (n + 1) % 2)
        seq += [(min(i), max(i)) for i in zip(wire_labels[::2], wire_labels_new[::2]) if i[0] != i[1]]
        wire_labels = wire_labels_new
    return seq