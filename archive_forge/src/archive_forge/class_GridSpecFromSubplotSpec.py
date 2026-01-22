import copy
import logging
from numbers import Integral
import numpy as np
import matplotlib as mpl
from matplotlib import _api, _pylab_helpers, _tight_layout
from matplotlib.transforms import Bbox
class GridSpecFromSubplotSpec(GridSpecBase):
    """
    GridSpec whose subplot layout parameters are inherited from the
    location specified by a given SubplotSpec.
    """

    def __init__(self, nrows, ncols, subplot_spec, wspace=None, hspace=None, height_ratios=None, width_ratios=None):
        """
        Parameters
        ----------
        nrows, ncols : int
            Number of rows and number of columns of the grid.
        subplot_spec : SubplotSpec
            Spec from which the layout parameters are inherited.
        wspace, hspace : float, optional
            See `GridSpec` for more details. If not specified default values
            (from the figure or rcParams) are used.
        height_ratios : array-like of length *nrows*, optional
            See `GridSpecBase` for details.
        width_ratios : array-like of length *ncols*, optional
            See `GridSpecBase` for details.
        """
        self._wspace = wspace
        self._hspace = hspace
        self._subplot_spec = subplot_spec
        self.figure = self._subplot_spec.get_gridspec().figure
        super().__init__(nrows, ncols, width_ratios=width_ratios, height_ratios=height_ratios)

    def get_subplot_params(self, figure=None):
        """Return a dictionary of subplot layout parameters."""
        hspace = self._hspace if self._hspace is not None else figure.subplotpars.hspace if figure is not None else mpl.rcParams['figure.subplot.hspace']
        wspace = self._wspace if self._wspace is not None else figure.subplotpars.wspace if figure is not None else mpl.rcParams['figure.subplot.wspace']
        figbox = self._subplot_spec.get_position(figure)
        left, bottom, right, top = figbox.extents
        return mpl.figure.SubplotParams(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)

    def get_topmost_subplotspec(self):
        """
        Return the topmost `.SubplotSpec` instance associated with the subplot.
        """
        return self._subplot_spec.get_topmost_subplotspec()