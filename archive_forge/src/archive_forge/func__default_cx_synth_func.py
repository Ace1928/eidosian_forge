from __future__ import annotations
from collections.abc import Callable
import numpy as np
from qiskit.circuit import QuantumCircuit
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Clifford  # pylint: disable=cyclic-import
from qiskit.quantum_info.operators.symplectic.clifford_circuits import (
from qiskit.synthesis.linear import (
from qiskit.synthesis.linear_phase import synth_cz_depth_line_mr, synth_cx_cz_depth_line_my
from qiskit.synthesis.linear.linear_matrix_utils import (
def _default_cx_synth_func(mat):
    """
    Construct the layer of CX gates from a boolean invertible matrix mat.
    """
    CX_circ = synth_cnot_count_full_pmh(mat)
    CX_circ.name = 'CX'
    return CX_circ