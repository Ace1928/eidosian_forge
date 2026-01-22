from typing import List, Union, Type, cast, TYPE_CHECKING
from enum import Enum
import numpy as np
from cirq import ops, transformers, protocols, linalg
from cirq.type_workarounds import NotImplementedType
def _matrix_to_clifford_op(mat: np.ndarray, qubit: 'cirq.Qid', *, atol: float) -> Union[ops.Operation, NotImplementedType]:
    rotations = transformers.single_qubit_matrix_to_pauli_rotations(mat, atol)
    clifford_gate = ops.SingleQubitCliffordGate.I
    for pauli, half_turns in rotations:
        if linalg.all_near_zero_mod(half_turns, 0.5):
            quarter_turns = round(half_turns * 2) % 4
            clifford_gate = clifford_gate.merged_with(ops.SingleQubitCliffordGate.from_pauli(pauli, sqrt=bool(quarter_turns % 2)) ** (1 - 2 * int(quarter_turns == 3)))
        else:
            return NotImplemented
    return clifford_gate(qubit)