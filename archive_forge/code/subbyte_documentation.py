from typing import Tuple, Union
import numpy as np
Unpack a single byte 4bitx2 to two 4 bit elements
    Args:
        x: Input data
        signed: boolean, whether to interpret as signed int4.
    Returns:
        A tuple of ndarrays containing int4 elements (sign-extended to int8/uint8)
    