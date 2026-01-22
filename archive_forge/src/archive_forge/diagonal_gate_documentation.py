from typing import (
import numpy as np
import sympy
from cirq import protocols, value
from cirq._compat import proper_repr
from cirq.ops import common_gates, raw_types, global_phase_op
Decompose the n-qubit diagonal gates into CNOT and Rz gates.

        A 3 qubits decomposition looks like
        0: ───────────────────────────────────X───Rz(6)───X───Rz(7)───X───Rz(5)───X───Rz(4)───
                                              │           │           │           │
        1: ───────────X───Rz(3)───X───Rz(2)───@───────────┼───────────@───────────┼───────────
                      │           │                       │                       │
        2: ───Rz(1)───@───────────@───────────────────────@───────────────────────@───────────

        where the angles in Rz gates are corresponding to the fast-walsh-Hadamard transform
        of diagonal_angles in the Gray Code order.

        For n qubits decomposition looks similar but with 2^n-1 Rz gates and 2^n-2 CNOT gates.

        The algorithm is implemented according to the paper:
            Welch, Jonathan, et al. "Efficient quantum circuits for diagonal unitaries without
            ancillas." New Journal of Physics 16.3 (2014): 033040.
            https://iopscience.iop.org/article/10.1088/1367-2630/16/3/033040/meta
        