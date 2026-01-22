from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import (
def _get_lower_triangular(n, mat, mat_inv):
    mat = mat.copy()
    mat_t = mat.copy()
    mat_inv_t = mat_inv.copy()
    cx_instructions_rows = []
    for i in reversed(range(0, n)):
        found_first = False
        for j in reversed(range(0, n)):
            if mat[i, j]:
                if not found_first:
                    found_first = True
                    first_j = j
                else:
                    _col_op(mat, j, first_j)
        for k in reversed(range(0, i)):
            if mat[k, first_j]:
                _row_op_update_instructions(cx_instructions_rows, mat, i, k)
    for inst in cx_instructions_rows:
        _row_op(mat_t, inst[0], inst[1])
        _col_op(mat_inv_t, inst[0], inst[1])
    return (mat_t, mat_inv_t)