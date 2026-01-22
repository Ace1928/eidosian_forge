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
def _get_packed_offsets(widths, total, sep, mode='fixed'):
    """
    Pack boxes specified by their *widths*.

    For simplicity of the description, the terminology used here assumes a
    horizontal layout, but the function works equally for a vertical layout.

    There are three packing *mode*\\s:

    - 'fixed': The elements are packed tight to the left with a spacing of
      *sep* in between. If *total* is *None* the returned total will be the
      right edge of the last box. A non-*None* total will be passed unchecked
      to the output. In particular this means that right edge of the last
      box may be further to the right than the returned total.

    - 'expand': Distribute the boxes with equal spacing so that the left edge
      of the first box is at 0, and the right edge of the last box is at
      *total*. The parameter *sep* is ignored in this mode. A total of *None*
      is accepted and considered equal to 1. The total is returned unchanged
      (except for the conversion *None* to 1). If the total is smaller than
      the sum of the widths, the laid out boxes will overlap.

    - 'equal': If *total* is given, the total space is divided in N equal
      ranges and each box is left-aligned within its subspace.
      Otherwise (*total* is *None*), *sep* must be provided and each box is
      left-aligned in its subspace of width ``(max(widths) + sep)``. The
      total width is then calculated to be ``N * (max(widths) + sep)``.

    Parameters
    ----------
    widths : list of float
        Widths of boxes to be packed.
    total : float or None
        Intended total length. *None* if not used.
    sep : float or None
        Spacing between boxes.
    mode : {'fixed', 'expand', 'equal'}
        The packing mode.

    Returns
    -------
    total : float
        The total width needed to accommodate the laid out boxes.
    offsets : array of float
        The left offsets of the boxes.
    """
    _api.check_in_list(['fixed', 'expand', 'equal'], mode=mode)
    if mode == 'fixed':
        offsets_ = np.cumsum([0] + [w + sep for w in widths])
        offsets = offsets_[:-1]
        if total is None:
            total = offsets_[-1] - sep
        return (total, offsets)
    elif mode == 'expand':
        if total is None:
            total = 1
        if len(widths) > 1:
            sep = (total - sum(widths)) / (len(widths) - 1)
        else:
            sep = 0
        offsets_ = np.cumsum([0] + [w + sep for w in widths])
        offsets = offsets_[:-1]
        return (total, offsets)
    elif mode == 'equal':
        maxh = max(widths)
        if total is None:
            if sep is None:
                raise ValueError("total and sep cannot both be None when using layout mode 'equal'")
            total = (maxh + sep) * len(widths)
        else:
            sep = total / len(widths) - maxh
        offsets = (maxh + sep) * np.arange(len(widths))
        return (total, offsets)