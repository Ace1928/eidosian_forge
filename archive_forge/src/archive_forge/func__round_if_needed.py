import numbers
import numpy
import cupy
def _round_if_needed(arr, dtype):
    """Rounds arr inplace if the destination dtype is an integer.
    """
    if cupy.issubdtype(dtype, cupy.integer):
        arr.round(out=arr)