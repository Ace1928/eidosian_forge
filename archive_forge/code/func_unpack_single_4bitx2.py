from typing import Tuple, Union
import numpy as np
def unpack_single_4bitx2(x: Union[np.ndarray, np.dtype, float], signed: bool) -> Tuple[np.ndarray, np.ndarray]:
    unpack_signed = lambda x: np.where(x >> 3 == 0, x, x | 240)
    'Unpack a single byte 4bitx2 to two 4 bit elements\n    Args:\n        x: Input data\n        signed: boolean, whether to interpret as signed int4.\n    Returns:\n        A tuple of ndarrays containing int4 elements (sign-extended to int8/uint8)\n    '
    if not isinstance(x, np.ndarray):
        x = np.asarray(x)
    x_low = x & 15
    x_high = x >> 4
    x_low = unpack_signed(x_low) if signed else x_low
    x_high = unpack_signed(x_high) if signed else x_high
    dtype = np.int8 if signed else np.uint8
    return (x_low.astype(dtype), x_high.astype(dtype))