from typing import List, Callable, TYPE_CHECKING
from scipy.linalg import cossin
import numpy as np
from cirq import ops
from cirq.linalg import decompositions, predicates
def _single_qubit_decomposition(qubit: 'cirq.Qid', u: np.ndarray) -> 'op_tree.OpTree':
    """Decomposes single-qubit gate, and returns list of operations, keeping phase invariant.

    Args:
        qubit: Qubit on which to apply operations
        u: (2 x 2) Numpy array for unitary representing 1-qubit gate to be decomposed

    Yields:
        A single operation from OP TREE of 3 operations (rz,ry,ZPowGate)
    """
    phi_0, phi_1, phi_2 = decompositions.deconstruct_single_qubit_matrix_into_angles(u)
    phase = np.angle(u[0, 0] / (np.exp(-1j * phi_0 / 2) * np.cos(phi_1 / 2)))
    yield ops.rz(phi_0).on(qubit)
    yield ops.ry(phi_1).on(qubit)
    yield ops.ZPowGate(exponent=phi_2 / np.pi, global_shift=phase / phi_2).on(qubit)