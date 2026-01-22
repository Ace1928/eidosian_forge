import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class nanmax(Stub):
    """nanmax(ary, idx, val)

        Perform atomic ``ary[idx] = max(ary[idx], val)``.

        NOTE: NaN is treated as a missing value such that:
        nanmax(NaN, n) == n, nanmax(n, NaN) == n

        Supported on int32, int64, uint32, uint64, float32, float64 operands
        only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """