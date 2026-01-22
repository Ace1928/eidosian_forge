from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import (
def _optimize_cx_circ_depth_5n_line(mat):
    mat_inv = mat.copy()
    mat_cpy = calc_inverse_matrix(mat_inv)
    n = len(mat_cpy)
    cx_instructions_rows_m2nw = _matrix_to_north_west(n, mat_cpy, mat_inv)
    cx_instructions_rows_nw2id = _north_west_to_identity(n, mat_cpy)
    return (cx_instructions_rows_m2nw, cx_instructions_rows_nw2id)