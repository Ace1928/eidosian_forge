import functools
import inspect
import sys
import warnings
import numpy as np
from ._warnings import all_warnings, warn
def _validate_interpolation_order(image_dtype, order):
    """Validate and return spline interpolation's order.

    Parameters
    ----------
    image_dtype : dtype
        Image dtype.
    order : int, optional
        The order of the spline interpolation. The order has to be in
        the range 0-5. See `skimage.transform.warp` for detail.

    Returns
    -------
    order : int
        if input order is None, returns 0 if image_dtype is bool and 1
        otherwise. Otherwise, image_dtype is checked and input order
        is validated accordingly (order > 0 is not supported for bool
        image dtype)

    """
    if order is None:
        return 0 if image_dtype == bool else 1
    if order < 0 or order > 5:
        raise ValueError('Spline interpolation order has to be in the range 0-5.')
    if image_dtype == bool and order != 0:
        raise ValueError('Input image dtype is bool. Interpolation is not defined with bool data type. Please set order to 0 or explicitly cast input image to another data type.')
    return order