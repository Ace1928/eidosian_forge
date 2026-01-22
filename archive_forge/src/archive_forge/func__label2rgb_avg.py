import itertools
import numpy as np
from .._shared.utils import _supported_float_type, warn
from ..util import img_as_float
from . import rgb_colors
from .colorconv import gray2rgb, rgb2hsv, hsv2rgb
def _label2rgb_avg(label_field, image, bg_label=0, bg_color=(0, 0, 0)):
    """Visualise each segment in `label_field` with its mean color in `image`.

    Parameters
    ----------
    label_field : ndarray of int
        A segmentation of an image.
    image : array, shape ``label_field.shape + (3,)``
        A color image of the same spatial shape as `label_field`.
    bg_label : int, optional
        A value in `label_field` to be treated as background.
    bg_color : 3-tuple of int, optional
        The color for the background label

    Returns
    -------
    out : ndarray, same shape and type as `image`
        The output visualization.
    """
    out = np.zeros(label_field.shape + (3,), dtype=image.dtype)
    labels = np.unique(label_field)
    bg = labels == bg_label
    if bg.any():
        labels = labels[labels != bg_label]
        mask = (label_field == bg_label).nonzero()
        out[mask] = bg_color
    for label in labels:
        mask = (label_field == label).nonzero()
        color = image[mask].mean(axis=0)
        out[mask] = color
    return out