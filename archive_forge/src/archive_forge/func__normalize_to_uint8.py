from typing import TYPE_CHECKING, Optional, Tuple, Union
def _normalize_to_uint8(x):
    is_int = np.issubdtype(x.dtype, np.integer)
    low = 0
    high = 255 if is_int else 1
    if x.min() < low or x.max() > high:
        if is_int:
            raise ValueError(f'Integer pixel values out of acceptable range [0, 255]. Found minimum value {x.min()} and maximum value {x.max()}. Ensure all pixel values are within the specified range.')
        else:
            raise ValueError(f'Float pixel values out of acceptable range [0.0, 1.0]. Found minimum value {x.min()} and maximum value {x.max()}. Ensure all pixel values are within the specified range.')
    if not is_int:
        x = x * 255
    return x.astype(np.uint8)