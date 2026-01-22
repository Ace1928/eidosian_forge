from __future__ import annotations
import numpy as np
from qiskit.exceptions import QiskitError
from qiskit.circuit import QuantumCircuit
from qiskit.synthesis.linear.linear_matrix_utils import (
def _row_op_update_instructions(cx_instructions, mat, a, b):
    cx_instructions.append((a, b))
    _row_op(mat, a, b)