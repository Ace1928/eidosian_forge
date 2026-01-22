import functools
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _docstring
import matplotlib.artist as martist
import matplotlib.path as mpath
import matplotlib.text as mtext
import matplotlib.transforms as mtransforms
from matplotlib.font_manager import FontProperties
from matplotlib.image import BboxImage
from matplotlib.patches import (
from matplotlib.transforms import Bbox, BboxBase, TransformedBbox
def _get_aligned_offsets(yspans, height, align='baseline'):
    """
    Align boxes each specified by their ``(y0, y1)`` spans.

    For simplicity of the description, the terminology used here assumes a
    horizontal layout (i.e., vertical alignment), but the function works
    equally for a vertical layout.

    Parameters
    ----------
    yspans
        List of (y0, y1) spans of boxes to be aligned.
    height : float or None
        Intended total height. If None, the maximum of the heights
        (``y1 - y0``) in *yspans* is used.
    align : {'baseline', 'left', 'top', 'right', 'bottom', 'center'}
        The alignment anchor of the boxes.

    Returns
    -------
    (y0, y1)
        y range spanned by the packing.  If a *height* was originally passed
        in, then for all alignments other than "baseline", a span of ``(0,
        height)`` is used without checking that it is actually large enough).
    descent
        The descent of the packing.
    offsets
        The bottom offsets of the boxes.
    """
    _api.check_in_list(['baseline', 'left', 'top', 'right', 'bottom', 'center'], align=align)
    if height is None:
        height = max((y1 - y0 for y0, y1 in yspans))
    if align == 'baseline':
        yspan = (min((y0 for y0, y1 in yspans)), max((y1 for y0, y1 in yspans)))
        offsets = [0] * len(yspans)
    elif align in ['left', 'bottom']:
        yspan = (0, height)
        offsets = [-y0 for y0, y1 in yspans]
    elif align in ['right', 'top']:
        yspan = (0, height)
        offsets = [height - y1 for y0, y1 in yspans]
    elif align == 'center':
        yspan = (0, height)
        offsets = [(height - (y1 - y0)) * 0.5 - y0 for y0, y1 in yspans]
    return (yspan, offsets)