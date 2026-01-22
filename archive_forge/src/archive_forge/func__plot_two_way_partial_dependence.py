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
def _plot_two_way_partial_dependence(self, avg_preds, feature_values, feature_idx, ax, pd_plot_idx, Z_level, contour_kw, categorical, heatmap_kw):
    """Plot 2-way partial dependence.

        Parameters
        ----------
        avg_preds : ndarray of shape                 (n_instances, n_grid_points, n_grid_points)
            The average predictions for all points of `feature_values[0]` and
            `feature_values[1]` for some given features for all samples in `X`.
        feature_values : seq of 1d array
            A sequence of array of the feature values for which the predictions
            have been computed.
        feature_idx : tuple of int
            The indices of the target features
        ax : Matplotlib axes
            The axis on which to plot the ICE and PDP lines.
        pd_plot_idx : int
            The sequential index of the plot. It will be unraveled to find the
            matching 2D position in the grid layout.
        Z_level : ndarray of shape (8, 8)
            The Z-level used to encode the average predictions.
        contour_kw : dict
            Dict with keywords passed when plotting the contours.
        categorical : bool
            Whether features are categorical.
        heatmap_kw: dict
            Dict with keywords passed when plotting the PD heatmap
            (categorical).
        """
    if categorical:
        import matplotlib.pyplot as plt
        default_im_kw = dict(interpolation='nearest', cmap='viridis')
        im_kw = {**default_im_kw, **heatmap_kw}
        data = avg_preds[self.target_idx]
        im = ax.imshow(data, **im_kw)
        text = None
        cmap_min, cmap_max = (im.cmap(0), im.cmap(1.0))
        text = np.empty_like(data, dtype=object)
        thresh = (data.max() + data.min()) / 2.0
        for flat_index in range(data.size):
            row, col = np.unravel_index(flat_index, data.shape)
            color = cmap_max if data[row, col] < thresh else cmap_min
            values_format = '.2f'
            text_data = format(data[row, col], values_format)
            text_kwargs = dict(ha='center', va='center', color=color)
            text[row, col] = ax.text(col, row, text_data, **text_kwargs)
        fig = ax.figure
        fig.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(len(feature_values[1])), yticks=np.arange(len(feature_values[0])), xticklabels=feature_values[1], yticklabels=feature_values[0], xlabel=self.feature_names[feature_idx[1]], ylabel=self.feature_names[feature_idx[0]])
        plt.setp(ax.get_xticklabels(), rotation='vertical')
        heatmap_idx = np.unravel_index(pd_plot_idx, self.heatmaps_.shape)
        self.heatmaps_[heatmap_idx] = im
    else:
        from matplotlib import transforms
        XX, YY = np.meshgrid(feature_values[0], feature_values[1])
        Z = avg_preds[self.target_idx].T
        CS = ax.contour(XX, YY, Z, levels=Z_level, linewidths=0.5, colors='k')
        contour_idx = np.unravel_index(pd_plot_idx, self.contours_.shape)
        self.contours_[contour_idx] = ax.contourf(XX, YY, Z, levels=Z_level, vmax=Z_level[-1], vmin=Z_level[0], **contour_kw)
        ax.clabel(CS, fmt='%2.2f', colors='k', fontsize=10, inline=True)
        trans = transforms.blended_transform_factory(ax.transData, ax.transAxes)
        xlim, ylim = (ax.get_xlim(), ax.get_ylim())
        vlines_idx = np.unravel_index(pd_plot_idx, self.deciles_vlines_.shape)
        self.deciles_vlines_[vlines_idx] = ax.vlines(self.deciles[feature_idx[0]], 0, 0.05, transform=trans, color='k')
        hlines_idx = np.unravel_index(pd_plot_idx, self.deciles_hlines_.shape)
        self.deciles_hlines_[hlines_idx] = ax.hlines(self.deciles[feature_idx[1]], 0, 0.05, transform=trans, color='k')
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        if not ax.get_xlabel():
            ax.set_xlabel(self.feature_names[feature_idx[0]])
        ax.set_ylabel(self.feature_names[feature_idx[1]])