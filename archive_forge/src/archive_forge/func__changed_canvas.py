from contextlib import ExitStack
import copy
import itertools
from numbers import Integral, Number
from cycler import cycler
import numpy as np
import matplotlib as mpl
from . import (_api, _docstring, backend_tools, cbook, collections, colors,
from .lines import Line2D
from .patches import Circle, Rectangle, Ellipse, Polygon
from .transforms import TransformedPatchPath, Affine2D
def _changed_canvas(self):
    """
        Someone has switched the canvas on us!

        This happens if `savefig` needs to save to a format the previous
        backend did not support (e.g. saving a figure using an Agg based
        backend saved to a vector format).

        Returns
        -------
        bool
           True if the canvas has been changed.

        """
    return self.canvas is not self.ax.figure.canvas