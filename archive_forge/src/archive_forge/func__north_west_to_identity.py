from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import (
def _north_west_to_identity(n, mat):
    label_arr = list(reversed(range(n)))
    first_qubit = 0
    empty_layers = 0
    done = False
    cx_instructions_rows = []
    while not done:
        at_least_one_needed = False
        for i in range(first_qubit, n - 1, 2):
            if label_arr[i] > label_arr[i + 1]:
                at_least_one_needed = True
                if not mat[i, label_arr[i + 1]]:
                    _row_op_update_instructions(cx_instructions_rows, mat, i + 1, i)
                _row_op_update_instructions(cx_instructions_rows, mat, i, i + 1)
                _row_op_update_instructions(cx_instructions_rows, mat, i + 1, i)
                label_arr[i], label_arr[i + 1] = (label_arr[i + 1], label_arr[i])
        if not at_least_one_needed:
            empty_layers += 1
            if empty_layers > 1:
                done = True
        else:
            empty_layers = 0
        first_qubit = int(not first_qubit)
    return cx_instructions_rows