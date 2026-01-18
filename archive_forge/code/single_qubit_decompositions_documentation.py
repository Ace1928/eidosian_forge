import math
from typing import List, Optional, Tuple
import numpy as np
import sympy
from cirq import ops, linalg, protocols
from cirq.linalg.tolerance import near_zero_mod
Implements a single-qubit operation with a PhasedXZ gate.

    Under the hood, this uses deconstruct_single_qubit_matrix_into_angles which
    converts the given matrix to a series of three rotations around the Z, Y, Z
    axes. This is then converted to a phased X rotation followed by a Z, in the
    form of a single PhasedXZ gate.

    Args:
        mat: The 2x2 unitary matrix of the operation to implement.
        atol: A limit on the amount of error introduced by the
            construction.

    Returns:
        A PhasedXZ gate that implements the given matrix, or None if it is
        close to identity (trace distance <= atol).
    