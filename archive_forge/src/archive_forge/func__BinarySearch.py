import array
import contextlib
import enum
import struct
def _BinarySearch(values, value, pred=lambda x, y: x < y):
    """Implementation of C++ std::binary_search() algorithm."""
    index = _LowerBound(values, value, pred)
    if index != len(values) and (not pred(value, values[index])):
        return index
    return -1