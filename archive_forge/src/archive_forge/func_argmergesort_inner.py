import numpy as np
from collections import namedtuple
@wrap(**kwargs_lite)
def argmergesort_inner(arr, vals, ws):
    """The actual mergesort function

        Parameters
        ----------
        arr : array [read+write]
            The values being sorted inplace.  For argsort, this is the
            indices.
        vals : array [readonly]
            ``None`` for normal sort.  In argsort, this is the actual array values.
        ws : array [write]
            The workspace.  Must be of size ``arr.size // 2``
        """
    if arr.size > SMALL_MERGESORT:
        mid = arr.size // 2
        argmergesort_inner(arr[:mid], vals, ws)
        argmergesort_inner(arr[mid:], vals, ws)
        for i in range(mid):
            ws[i] = arr[i]
        left = ws[:mid]
        right = arr[mid:]
        out = arr
        i = j = k = 0
        while i < left.size and j < right.size:
            if not lessthan(right[j], left[i], vals):
                out[k] = left[i]
                i += 1
            else:
                out[k] = right[j]
                j += 1
            k += 1
        while i < left.size:
            out[k] = left[i]
            i += 1
            k += 1
        while j < right.size:
            out[k] = right[j]
            j += 1
            k += 1
    else:
        i = 1
        while i < arr.size:
            j = i
            while j > 0 and lessthan(arr[j], arr[j - 1], vals):
                arr[j - 1], arr[j] = (arr[j], arr[j - 1])
                j -= 1
            i += 1