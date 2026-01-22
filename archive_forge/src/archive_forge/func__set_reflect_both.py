import numbers
import numpy
import cupy
def _set_reflect_both(padded, axis, width_pair, method, include_edge=False):
    """Pads an `axis` of `arr` using reflection.

    Args:
      padded(cupy.ndarray): Input array of arbitrary shape.
      axis(int): Axis along which to pad `arr`.
      width_pair((int, int)): Pair of widths that mark the pad area on both
          sides in the given dimension.
      method(str): Controls method of reflection; options are 'even' or 'odd'.
      include_edge(bool, optional): If true, edge value is included in
          reflection, otherwise the edge value forms the symmetric axis to the
          reflection. (Default value = False)
    """
    left_pad, right_pad = width_pair
    old_length = padded.shape[axis] - right_pad - left_pad
    if include_edge:
        edge_offset = 1
    else:
        edge_offset = 0
        old_length -= 1
    if left_pad > 0:
        chunk_length = min(old_length, left_pad)
        stop = left_pad - edge_offset
        start = stop + chunk_length
        left_slice = _slice_at_axis(slice(start, stop, -1), axis)
        left_chunk = padded[left_slice]
        if method == 'odd':
            edge_slice = _slice_at_axis(slice(left_pad, left_pad + 1), axis)
            left_chunk = 2 * padded[edge_slice] - left_chunk
        start = left_pad - chunk_length
        stop = left_pad
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = left_chunk
        left_pad -= chunk_length
    if right_pad > 0:
        chunk_length = min(old_length, right_pad)
        start = -right_pad + edge_offset - 2
        stop = start - chunk_length
        right_slice = _slice_at_axis(slice(start, stop, -1), axis)
        right_chunk = padded[right_slice]
        if method == 'odd':
            edge_slice = _slice_at_axis(slice(-right_pad - 1, -right_pad), axis)
            right_chunk = 2 * padded[edge_slice] - right_chunk
        start = padded.shape[axis] - right_pad
        stop = start + chunk_length
        pad_area = _slice_at_axis(slice(start, stop), axis)
        padded[pad_area] = right_chunk
        right_pad -= chunk_length
    return (left_pad, right_pad)