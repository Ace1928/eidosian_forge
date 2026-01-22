import abc
import collections
import io
from typing import (
import numpy as np
import pandas as pd
from cirq import value, ops
from cirq._compat import proper_repr
from cirq.study import resolver
def _pack_bits(bits: np.ndarray) -> str:
    return np.packbits(bits).tobytes().hex()