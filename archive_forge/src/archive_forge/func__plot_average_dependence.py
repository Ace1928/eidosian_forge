import numbers
from itertools import chain
from math import ceil
import numpy as np
from scipy import sparse
from scipy.stats.mstats import mquantiles
from ...base import is_regressor
from ...utils import (
from ...utils._encode import _unique
from ...utils.parallel import Parallel, delayed
from .. import partial_dependence
from .._pd_utils import _check_feature_names, _get_feature_index
def _plot_average_dependence(self, avg_preds, feature_values, ax, pd_line_idx, line_kw, categorical, bar_kw):
    """Plot the average partial dependence.

        Parameters
        ----------
        avg_preds : ndarray of shape (n_grid_points,)
            The average predictions for all points of `feature_values` for a
            given feature for all samples in `X`.
        feature_values : ndarray of shape (n_grid_points,)
            The feature values for which the predictions have been computed.
        ax : Matplotlib axes
            The axis on which to plot the average PD.
        pd_line_idx : int
            The sequential index of the plot. It will be unraveled to find the
            matching 2D position in the grid layout.
        line_kw : dict
            Dict with keywords passed when plotting the PD plot.
        categorical : bool
            Whether feature is categorical.
        bar_kw: dict
            Dict with keywords passed when plotting the PD bars (categorical).
        """
    if categorical:
        bar_idx = np.unravel_index(pd_line_idx, self.bars_.shape)
        self.bars_[bar_idx] = ax.bar(feature_values, avg_preds, **bar_kw)[0]
        ax.tick_params(axis='x', rotation=90)
    else:
        line_idx = np.unravel_index(pd_line_idx, self.lines_.shape)
        self.lines_[line_idx] = ax.plot(feature_values, avg_preds, **line_kw)[0]