from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.linalg import is_unitary, is_special_unitary, map_eigenvalues
from cirq.protocols import unitary
def _decompose_single_ctrl(matrix: np.ndarray, control: 'cirq.Qid', target: 'cirq.Qid') -> List['cirq.Operation']:
    """Decomposes controlled gate with one control.

    See [1], chapter 5.1.
    """
    a, b, c, delta = _decompose_abc(matrix)
    result = [ops.ZPowGate(exponent=delta / np.pi).on(control), ops.MatrixGate(c).on(target), ops.CNOT.on(control, target), ops.MatrixGate(b).on(target), ops.CNOT.on(control, target), ops.MatrixGate(a).on(target)]
    result = [g for g in result if not _is_identity(unitary(g))]
    return result