from typing import List, Tuple, TYPE_CHECKING
import numpy as np
from cirq import ops
from cirq.linalg import is_unitary, is_special_unitary, map_eigenvalues
from cirq.protocols import unitary
def _decompose_recursive(matrix: np.ndarray, power: float, controls: List['cirq.Qid'], target: 'cirq.Qid', free_qubits: List['cirq.Qid']) -> List['cirq.Operation']:
    """Decomposes controlled unitary gate into elementary gates.

    Result has O(len(controls)^2) operations.
    See [1], lemma 7.5.
    """
    if len(controls) == 1:
        return _decompose_single_ctrl(_unitary_power(matrix, power), controls[0], target)
    cnots = decompose_multi_controlled_x(controls[:-1], controls[-1], free_qubits + [target])
    return [*_decompose_single_ctrl(_unitary_power(matrix, 0.5 * power), controls[-1], target), *cnots, *_decompose_single_ctrl(_unitary_power(matrix, -0.5 * power), controls[-1], target), *cnots, *_decompose_recursive(matrix, 0.5 * power, controls[:-1], target, [controls[-1]] + free_qubits)]