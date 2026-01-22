import numpy as np
from collections import defaultdict
import functools
import itertools
from inspect import Signature, Parameter
class exch(Stub):
    """exch(ary, idx, val)

        Perform atomic ``ary[idx] = val``. Supported on int32, int64, uint32 and
        uint64 operands only.

        Returns the old value at the index location as if it is loaded
        atomically.
        """