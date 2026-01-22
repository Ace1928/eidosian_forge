import copy
from collections.abc import Sized
import numpy as np
import matplotlib as mpl
from . import _api, cbook
from .path import Path
from .transforms import IdentityTransform, Affine2D
from ._enums import JoinStyle, CapStyle
def _set_fillstyle(self, fillstyle):
    """
        Set the fillstyle.

        Parameters
        ----------
        fillstyle : {'full', 'left', 'right', 'bottom', 'top', 'none'}
            The part of the marker surface that is colored with
            markerfacecolor.
        """
    if fillstyle is None:
        fillstyle = mpl.rcParams['markers.fillstyle']
    _api.check_in_list(self.fillstyles, fillstyle=fillstyle)
    self._fillstyle = fillstyle