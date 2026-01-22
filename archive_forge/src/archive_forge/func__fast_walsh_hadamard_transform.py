from typing import Optional, Tuple, cast
import numpy as np
import numpy.typing as npt
from cirq.ops import DensePauliString
from cirq import protocols
def _fast_walsh_hadamard_transform(V: npt.NDArray) -> None:
    """Fast Walsh–Hadamard Transform of an array."""
    m = len(V)
    n = m.bit_length() - 1
    for h in [2 ** i for i in range(n)]:
        for i in range(0, m, h * 2):
            for j in range(i, i + h):
                x = V[j]
                y = V[j + h]
                V[j] = x + y
                V[j + h] = x - y