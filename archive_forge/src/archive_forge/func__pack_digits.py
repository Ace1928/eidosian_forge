import abc
import collections
import io
from typing import (
import numpy as np
import pandas as pd
from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver
def _pack_digits(digits: np.ndarray, pack_bits: str='auto') -> Tuple[str, bool]:
    """Returns a string of packed digits and a boolean indicating whether the
    digits were packed as binary values.

    Args:
        digits: A numpy array.
        pack_bits: If 'auto' (the default), automatically pack binary digits
            using `np.packbits` to save space. If 'never', do not pack binary
            digits. If 'force', use `np.packbits` without checking for
            compatibility.

    Raises:
        ValueError: If `pack_bits` is not `auto`, `force`, or `never`.
    """
    if pack_bits == 'force':
        return (_pack_bits(digits), True)
    if pack_bits not in ['auto', 'never']:
        raise ValueError("Please set `pack_bits` to 'auto', 'force', or 'never'.")
    if pack_bits == 'auto' and np.array_equal(digits, digits.astype(np.bool_)):
        return (_pack_bits(digits.astype(np.bool_)), True)
    buffer = io.BytesIO()
    np.save(buffer, digits, allow_pickle=False)
    buffer.seek(0)
    packed_digits = buffer.read().hex()
    buffer.close()
    return (packed_digits, False)