from __future__ import absolute_import, division, print_function
import sys
import math
import struct
import numpy as np
import warnings
def write_variable_length(value):
    """
    Write a variable length variable.

    Parameters
    ----------
    value : bytearray
        Value to be encoded as a variable of variable length.

    Returns
    -------
    bytearray
        Variable with variable length.

    """
    result = bytearray()
    result.insert(0, value & 127)
    value >>= 7
    if value:
        result.insert(0, value & 127 | 128)
        value >>= 7
        if value:
            result.insert(0, value & 127 | 128)
            value >>= 7
            if value:
                result.insert(0, value & 127 | 128)
    return result