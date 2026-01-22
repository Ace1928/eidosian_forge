from contextlib import ExitStack
import inspect
import itertools
import logging
from numbers import Integral
import threading
import numpy as np
import matplotlib as mpl
from matplotlib import _blocking_input, backend_bases, _docstring, projections
from matplotlib.artist import (
from matplotlib.backend_bases import (
import matplotlib._api as _api
import matplotlib.cbook as cbook
import matplotlib.colorbar as cbar
import matplotlib.image as mimage
from matplotlib.axes import Axes
from matplotlib.gridspec import GridSpec
from matplotlib.layout_engine import (
import matplotlib.legend as mlegend
from matplotlib.patches import Rectangle
from matplotlib.text import Text
from matplotlib.transforms import (Affine2D, Bbox, BboxTransformTo,
class SubplotParams:
    """
    A class to hold the parameters for a subplot.
    """

    def __init__(self, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        """
        Defaults are given by :rc:`figure.subplot.[name]`.

        Parameters
        ----------
        left : float
            The position of the left edge of the subplots,
            as a fraction of the figure width.
        right : float
            The position of the right edge of the subplots,
            as a fraction of the figure width.
        bottom : float
            The position of the bottom edge of the subplots,
            as a fraction of the figure height.
        top : float
            The position of the top edge of the subplots,
            as a fraction of the figure height.
        wspace : float
            The width of the padding between subplots,
            as a fraction of the average Axes width.
        hspace : float
            The height of the padding between subplots,
            as a fraction of the average Axes height.
        """
        for key in ['left', 'bottom', 'right', 'top', 'wspace', 'hspace']:
            setattr(self, key, mpl.rcParams[f'figure.subplot.{key}'])
        self.update(left, bottom, right, top, wspace, hspace)

    def update(self, left=None, bottom=None, right=None, top=None, wspace=None, hspace=None):
        """
        Update the dimensions of the passed parameters. *None* means unchanged.
        """
        if (left if left is not None else self.left) >= (right if right is not None else self.right):
            raise ValueError('left cannot be >= right')
        if (bottom if bottom is not None else self.bottom) >= (top if top is not None else self.top):
            raise ValueError('bottom cannot be >= top')
        if left is not None:
            self.left = left
        if right is not None:
            self.right = right
        if bottom is not None:
            self.bottom = bottom
        if top is not None:
            self.top = top
        if wspace is not None:
            self.wspace = wspace
        if hspace is not None:
            self.hspace = hspace