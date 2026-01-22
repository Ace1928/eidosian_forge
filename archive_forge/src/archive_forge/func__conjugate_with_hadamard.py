from typing import Optional, Tuple, cast
import numpy as np
import numpy.typing as npt
from cirq.ops import DensePauliString
from cirq import protocols
def _conjugate_with_hadamard(U: npt.NDArray) -> npt.NDArray:
    """Applies HcUH in O(n4^n) instead of O(8^n)."""
    U = np.copy(U.T)
    for i in range(U.shape[1]):
        _fast_walsh_hadamard_transform(U[:, i])
    U = U.T
    for i in range(U.shape[1]):
        _fast_walsh_hadamard_transform(U[:, i])
    return U