import functools
import logging
import math
from numbers import Real
import weakref
import numpy as np
import matplotlib as mpl
from . import _api, artist, cbook, _docstring
from .artist import Artist
from .font_manager import FontProperties
from .patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from .textpath import TextPath, TextToPath  # noqa # Logically located here
from .transforms import (
def _get_textbox(text, renderer):
    """
    Calculate the bounding box of the text.

    The bbox position takes text rotation into account, but the width and
    height are those of the unrotated box (unlike `.Text.get_window_extent`).
    """
    projected_xs = []
    projected_ys = []
    theta = np.deg2rad(text.get_rotation())
    tr = Affine2D().rotate(-theta)
    _, parts, d = text._get_layout(renderer)
    for t, wh, x, y in parts:
        w, h = wh
        xt1, yt1 = tr.transform((x, y))
        yt1 -= d
        xt2, yt2 = (xt1 + w, yt1 + h)
        projected_xs.extend([xt1, xt2])
        projected_ys.extend([yt1, yt2])
    xt_box, yt_box = (min(projected_xs), min(projected_ys))
    w_box, h_box = (max(projected_xs) - xt_box, max(projected_ys) - yt_box)
    x_box, y_box = Affine2D().rotate(theta).transform((xt_box, yt_box))
    return (x_box, y_box, w_box, h_box)