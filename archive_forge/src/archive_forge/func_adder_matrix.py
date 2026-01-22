from typing import Union, Sequence
import pytest
import numpy as np
import cirq
def adder_matrix(target_width: int, source_width: int) -> np.ndarray:
    t, s = (target_width, source_width)
    result = np.zeros((t, s, t, s))
    for k in range(s):
        result[:, k, :, k] = shift_matrix(t, k)
    result.shape = (t * s, t * s)
    return result