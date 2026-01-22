import abc
import collections
import io
from typing import (
import numpy as np
import pandas as pd
from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver
def _unpack_digits(packed_digits: str, binary: bool, dtype: Union[None, str], shape: Union[None, Sequence[int]]) -> np.ndarray:
    """The opposite of `_pack_digits`.

    Args:
        packed_digits: The hex-encoded string representing a numpy array of
            digits. This is the first return value of `_pack_digits`.
        binary: Whether the digits have been packed as binary. This is the
            second return value of `_pack_digits`.
        dtype: If `binary` is True, you must also provide the datatype of the
            array. Otherwise, dtype information is contained within the hex
            string.
        shape: If `binary` is True, you must also provide the shape of the
            array. Otherwise, shape information is contained within the hex
            string.
    """
    if binary:
        dtype = cast(str, dtype)
        shape = cast(Sequence[int], shape)
        return _unpack_bits(packed_digits, dtype, shape)
    buffer = io.BytesIO()
    buffer.write(bytes.fromhex(packed_digits))
    buffer.seek(0)
    digits = np.load(buffer, allow_pickle=False)
    buffer.close()
    return digits