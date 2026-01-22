from typing import Optional, Sequence, Tuple, TYPE_CHECKING
import numpy as np
import sympy
from cirq import circuits, ops, linalg, protocols
from cirq.transformers.analytical_decompositions import single_qubit_decompositions
from cirq.transformers.merge_single_qubit_gates import merge_single_qubit_gates_to_phxz
def _in_2sqrt_iswap_region(interaction_coefficients: Tuple[float, float, float], weyl_tol: float=1e-08) -> bool:
    """Tests if (x, y, z) is inside or within weyl_tol of the volume
    x >= y + |z| assuming x, y, z are canonical.

    References:
        Towards ultra-high fidelity quantum operations: SQiSW gate as a native
        two-qubit gate
        https://arxiv.org/abs/2105.06074
    """
    x, y, z = interaction_coefficients
    return x + weyl_tol >= y + abs(z)