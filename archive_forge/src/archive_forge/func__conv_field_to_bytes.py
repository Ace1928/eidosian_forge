import collections
import gzip
import io
import logging
import struct
import numpy as np
def _conv_field_to_bytes(field_value, field_type):
    """
    Auxiliary function that converts `field_value` to bytes based on request `field_type`,
    for saving to the binary file.

    Parameters
    ----------
    field_value: numerical
        contains arguments of the string and start/end indexes of the bad portion.

    field_type: str
        currently supported `field_types` are `i` for 32-bit integer and `d` for 64-bit float
    """
    if field_type == 'i':
        return np.int32(field_value).tobytes()
    elif field_type == 'd':
        return np.float64(field_value).tobytes()
    else:
        raise NotImplementedError('Currently conversion to "%s" type is not implemmented.' % field_type)