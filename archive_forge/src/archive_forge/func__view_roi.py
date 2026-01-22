import numbers
import numpy
import cupy
def _view_roi(array, original_area_slice, axis):
    """Gets a view of the current region of interest during iterative padding.

    When padding multiple dimensions iteratively corner values are
    unnecessarily overwritten multiple times. This function reduces the
    working area for the first dimensions so that corners are excluded.

    Args:
      array(cupy.ndarray): The array with the region of interest.
      original_area_slice(tuple of slices): Denotes the area with original
          values of the unpadded array.
      axis(int): The currently padded dimension assuming that `axis` is padded
          before `axis` + 1.

    Returns:
    """
    axis += 1
    sl = (slice(None),) * axis + original_area_slice[axis:]
    return array[sl]