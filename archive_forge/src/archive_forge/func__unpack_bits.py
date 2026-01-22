import abc
import collections
import io
from typing import (
import numpy as np
import pandas as pd
from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver
def _unpack_bits(packed_bits: str, dtype: str, shape: Sequence[int]) -> np.ndarray:
    bits_bytes = bytes.fromhex(packed_bits)
    bits = np.unpackbits(np.frombuffer(bits_bytes, dtype=np.uint8))
    return bits[:np.prod(shape).item()].reshape(shape).astype(dtype)