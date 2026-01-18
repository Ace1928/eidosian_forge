import collections
import numpy as np
from numba.core import types

        Three-way partition [low, high) around a chosen pivot.
        A tuple (lt, gt) is returned such that:
            - all elements in [low, lt) are < pivot
            - all elements in [lt, gt] are == pivot
            - all elements in (gt, high] are > pivot
        