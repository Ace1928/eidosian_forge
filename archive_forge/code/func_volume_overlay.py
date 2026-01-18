from __future__ import (absolute_import, division, print_function,
import numpy as np
from matplotlib import colors as mcolors
from matplotlib.collections import LineCollection, PolyCollection
from matplotlib.lines import TICKLEFT, TICKRIGHT, Line2D
from matplotlib.patches import Rectangle
from matplotlib.transforms import Affine2D
from six.moves import xrange, zip
def volume_overlay(ax, opens, closes, volumes, colorup='k', colordown='r', width=4, alpha=1.0):
    """Add a volume overlay to the current axes.  The opens and closes
    are used to determine the color of the bar.  -1 is missing.  If a
    value is missing on one it must be missing on all

    Parameters
    ----------
    ax : `Axes`
        an Axes instance to plot to
    opens : sequence
        a sequence of opens
    closes : sequence
        a sequence of closes
    volumes : sequence
        a sequence of volumes
    width : int
        the bar width in points
    colorup : color
        the color of the lines where close >= open
    colordown : color
        the color of the lines where close <  open
    alpha : float
        bar transparency

    Returns
    -------
    ret : `barCollection`
        The `barrCollection` added to the axes

    """
    colorup = mcolors.to_rgba(colorup, alpha)
    colordown = mcolors.to_rgba(colordown, alpha)
    colord = {True: colorup, False: colordown}
    colors = [colord[open < close] for open, close in zip(opens, closes) if open != -1 and close != -1]
    delta = width / 2.0
    bars = [((i - delta, 0), (i - delta, v), (i + delta, v), (i + delta, 0)) for i, v in enumerate(volumes) if v != -1]
    barCollection = PolyCollection(bars, facecolors=colors, edgecolors=((0, 0, 0, 1),), antialiaseds=(0,), linewidths=(0.5,))
    ax.add_collection(barCollection)
    corners = ((0, 0), (len(bars), max(volumes)))
    ax.update_datalim(corners)
    ax.autoscale_view()
    return barCollection