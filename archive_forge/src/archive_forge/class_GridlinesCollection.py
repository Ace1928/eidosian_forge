from operator import methodcaller
import numpy as np
import matplotlib as mpl
from matplotlib import _api, cbook
import matplotlib.artist as martist
import matplotlib.colors as mcolors
import matplotlib.text as mtext
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from matplotlib.patches import PathPatch
from matplotlib.path import Path
from matplotlib.transforms import (
from .axisline_style import AxislineStyle
class GridlinesCollection(LineCollection):

    def __init__(self, *args, which='major', axis='both', **kwargs):
        """
        Collection of grid lines.

        Parameters
        ----------
        which : {"major", "minor"}
            Which grid to consider.
        axis : {"both", "x", "y"}
            Which axis to consider.
        *args, **kwargs
            Passed to `.LineCollection`.
        """
        self._which = which
        self._axis = axis
        super().__init__(*args, **kwargs)
        self.set_grid_helper(None)

    def set_which(self, which):
        """
        Select major or minor grid lines.

        Parameters
        ----------
        which : {"major", "minor"}
        """
        self._which = which

    def set_axis(self, axis):
        """
        Select axis.

        Parameters
        ----------
        axis : {"both", "x", "y"}
        """
        self._axis = axis

    def set_grid_helper(self, grid_helper):
        """
        Set grid helper.

        Parameters
        ----------
        grid_helper : `.GridHelperBase` subclass
        """
        self._grid_helper = grid_helper

    def draw(self, renderer):
        if self._grid_helper is not None:
            self._grid_helper.update_lim(self.axes)
            gl = self._grid_helper.get_gridlines(self._which, self._axis)
            self.set_segments([np.transpose(l) for l in gl])
        super().draw(renderer)