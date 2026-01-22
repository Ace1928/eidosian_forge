import numbers
import numpy
import cupy
def _slice_at_axis(sl, axis):
    """Constructs a tuple of slices to slice an array in the given dimension.

    Args:
      sl(slice): The slice for the given dimension.
      axis(int): The axis to which `sl` is applied. All other dimensions are
          left "unsliced".

    Returns:
      tuple of slices: A tuple with slices matching `shape` in length.
    """
    return (slice(None),) * axis + (sl,) + (Ellipsis,)