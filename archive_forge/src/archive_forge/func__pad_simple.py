import numbers
import numpy
import cupy
def _pad_simple(array, pad_width, fill_value=None):
    """Pads an array on all sides with either a constant or undefined values.

    Args:
      array(cupy.ndarray): Array to grow.
      pad_width(sequence of tuple[int, int]): Pad width on both sides for each
          dimension in `arr`.
      fill_value(scalar, optional): If provided the padded area is
          filled with this value, otherwise the pad area left undefined.
          (Default value = None)
    """
    new_shape = tuple((left + size + right for size, (left, right) in zip(array.shape, pad_width)))
    order = 'F' if array.flags.fnc else 'C'
    padded = cupy.empty(new_shape, dtype=array.dtype, order=order)
    if fill_value is not None:
        padded.fill(fill_value)
    original_area_slice = tuple((slice(left, left + size) for size, (left, right) in zip(array.shape, pad_width)))
    padded[original_area_slice] = array
    return (padded, original_area_slice)