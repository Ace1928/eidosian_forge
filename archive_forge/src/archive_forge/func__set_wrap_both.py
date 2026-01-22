import numbers
import numpy
import cupy
def _set_wrap_both(padded, axis, width_pair):
    """Pads an `axis` of `arr` with wrapped values.

    Args:
      padded(cupy.ndarray): Input array of arbitrary shape.
      axis(int): Axis along which to pad `arr`.
      width_pair((int, int)): Pair of widths that mark the pad area on both
          sides in the given dimension.
    """
    left_pad, right_pad = width_pair
    period = padded.shape[axis] - right_pad - left_pad
    new_left_pad = 0
    new_right_pad = 0
    if left_pad > 0:
        right_slice = _slice_at_axis(slice(-right_pad - min(period, left_pad), -right_pad if right_pad != 0 else None), axis)
        right_chunk = padded[right_slice]
        if left_pad > period:
            pad_area = _slice_at_axis(slice(left_pad - period, left_pad), axis)
            new_left_pad = left_pad - period
        else:
            pad_area = _slice_at_axis(slice(None, left_pad), axis)
        padded[pad_area] = right_chunk
    if right_pad > 0:
        left_slice = _slice_at_axis(slice(left_pad, left_pad + min(period, right_pad)), axis)
        left_chunk = padded[left_slice]
        if right_pad > period:
            pad_area = _slice_at_axis(slice(-right_pad, -right_pad + period), axis)
            new_right_pad = right_pad - period
        else:
            pad_area = _slice_at_axis(slice(-right_pad, None), axis)
        padded[pad_area] = left_chunk
    return (new_left_pad, new_right_pad)