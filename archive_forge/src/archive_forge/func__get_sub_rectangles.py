import logging
import numpy as np
from ..core import Format, image_as_uint
from ._freeimage import fi, IO_FLAGS
from .freeimage import FreeimageFormat
def _get_sub_rectangles(self, prev, im):
    """
            Calculate the minimal rectangles that need updating each frame.
            Returns a two-element tuple containing the cropped images and a
            list of x-y positions.
            """
    diff = np.abs(im - prev)
    if diff.ndim == 3:
        diff = diff.sum(2)
    X = np.argwhere(diff.sum(0))
    Y = np.argwhere(diff.sum(1))
    if X.size and Y.size:
        x0, x1 = (int(X[0]), int(X[-1]) + 1)
        y0, y1 = (int(Y[0]), int(Y[-1]) + 1)
    else:
        x0, x1 = (0, 2)
        y0, y1 = (0, 2)
    return (im[y0:y1, x0:x1], (x0, y0))