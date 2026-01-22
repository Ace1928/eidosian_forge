from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.linalg import is_unitary, is_special_unitary, map_eigenvalues
from cirq.protocols import unitary
def _ccnot_congruent(c0: 'cirq.Qid', c1: 'cirq.Qid', target: 'cirq.Qid') -> List['cirq.Operation']:
    """Implements 3-qubit gate 'congruent' to CCNOT.

    Returns sequence of operations which is equivalent to applying
    CCNOT(c0, c1, target) and multiplying phase of |101> sate by -1.
    See lemma 6.2 in [1]."""
    return [ops.ry(-np.pi / 4).on(target), ops.CNOT(c1, target), ops.ry(-np.pi / 4).on(target), ops.CNOT(c0, target), ops.ry(np.pi / 4).on(target), ops.CNOT(c1, target), ops.ry(np.pi / 4).on(target)]