from copy import copy
from typing import Tuple
import numpy as np
import numpy.linalg as npl
import pennylane as qml
from pennylane.operation import Operation, Operator
from pennylane.wires import Wires
from pennylane import math
def _bisect_compute_a(u: np.ndarray):
    """
    Given the U matrix, compute the A matrix such that
    At x A x At x A x = U
    where At is the adjoint of A
    and x is the Pauli X matrix.
    """
    x = np.real(u[0, 1])
    z = u[1, 1]
    zr = np.real(z)
    zi = np.imag(z)
    if np.isclose(zr, -1):
        return np.array([[1, -1], [1, 1]]) * 2 ** (-0.5)
    ar = np.sqrt((np.sqrt((zr + 1) / 2) + 1) / 2)
    mul = 1 / (2 * np.sqrt((zr + 1) * (np.sqrt((zr + 1) / 2) + 1)))
    ai = zi * mul
    br = x * mul
    bi = 0
    return _param_su2(ar, ai, br, bi)