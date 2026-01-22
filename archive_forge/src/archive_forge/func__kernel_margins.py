from __future__ import absolute_import, division, print_function
import numpy as np
from .activations import linear, sigmoid, tanh
def _kernel_margins(kernel_shape, margin_shift):
    """
    Determine the margin that needs to be cut off when doing a "valid"
    convolution.

    Parameters
    ----------
    kernel_shape : tuple
        Shape of the convolution kernel to determine the margins for
    margin_shift : bool
        Shift the borders by one pixel if kernel is of even size

    Returns
    -------
    start_x, end_x, start_y, end_y : tuple
        Indices determining the valid part of the convolution output.
    """
    start_x = int(np.floor(kernel_shape[0] / 2.0))
    start_y = int(np.floor(kernel_shape[1] / 2.0))
    margin_shift = -1 if margin_shift else 0
    if kernel_shape[0] % 2 == 0:
        end_x = start_x - 1
        start_x += margin_shift
        end_x -= margin_shift
    else:
        end_x = start_x
    start_x = start_x if start_x > 0 else None
    end_x = -end_x if end_x > 0 else None
    if kernel_shape[1] % 2 == 0:
        end_y = start_y - 1
        start_y += margin_shift
        end_y -= margin_shift
    else:
        end_y = start_y
    start_y = start_y if start_y > 0 else None
    end_y = -end_y if end_y > 0 else None
    return (start_x, end_x, start_y, end_y)