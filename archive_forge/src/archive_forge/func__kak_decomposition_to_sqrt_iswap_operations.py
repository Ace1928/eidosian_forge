from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _kak_decomposition_to_sqrt_iswap_operations(q0: 'cirq.Qid', q1: 'cirq.Qid', kak: linalg.KakDecomposition, required_sqrt_iswap_count: Optional[int]=None, use_sqrt_iswap_inv: bool=False, atol: float=1e-08) -> Sequence['cirq.Operation']:
    single_qubit_operations, _ = _single_qubit_matrices_with_sqrt_iswap(kak, required_sqrt_iswap_count, atol=atol)
    if use_sqrt_iswap_inv:
        z_unitary = protocols.unitary(ops.Z)
        return _decomp_to_operations(q0, q1, ops.SQRT_ISWAP_INV, single_qubit_operations, u0_before=z_unitary, u0_after=z_unitary, atol=atol)
    return _decomp_to_operations(q0, q1, ops.SQRT_ISWAP, single_qubit_operations, atol=atol)