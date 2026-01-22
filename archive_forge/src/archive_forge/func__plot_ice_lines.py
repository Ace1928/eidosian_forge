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
def _plot_ice_lines(self, preds, feature_values, n_ice_to_plot, ax, pd_plot_idx, n_total_lines_by_plot, individual_line_kw):
    """Plot the ICE lines.

        Parameters
        ----------
        preds : ndarray of shape                 (n_instances, n_grid_points)
            The predictions computed for all points of `feature_values` for a
            given feature for all samples in `X`.
        feature_values : ndarray of shape (n_grid_points,)
            The feature values for which the predictions have been computed.
        n_ice_to_plot : int
            The number of ICE lines to plot.
        ax : Matplotlib axes
            The axis on which to plot the ICE lines.
        pd_plot_idx : int
            The sequential index of the plot. It will be unraveled to find the
            matching 2D position in the grid layout.
        n_total_lines_by_plot : int
            The total number of lines expected to be plot on the axis.
        individual_line_kw : dict
            Dict with keywords passed when plotting the ICE lines.
        """
    rng = check_random_state(self.random_state)
    ice_lines_idx = rng.choice(preds.shape[0], n_ice_to_plot, replace=False)
    ice_lines_subsampled = preds[ice_lines_idx, :]
    for ice_idx, ice in enumerate(ice_lines_subsampled):
        line_idx = np.unravel_index(pd_plot_idx * n_total_lines_by_plot + ice_idx, self.lines_.shape)
        self.lines_[line_idx] = ax.plot(feature_values, ice.ravel(), **individual_line_kw)[0]