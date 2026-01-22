from collections import namedtuple
from textwrap import dedent
import warnings
from colorsys import rgb_to_hls
from functools import partial
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import PatchCollection
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from seaborn._core.typing import default, deprecated
from seaborn._base import VectorPlotter, infer_orient, categorical_order
from seaborn._stats.density import KDE
from seaborn import utils
from seaborn.utils import (
from seaborn._compat import groupby_apply_include_groups
from seaborn._statistics import (
from seaborn.palettes import light_palette
from seaborn.axisgrid import FacetGrid, _facet_docs
class Beeswarm:
    """Modifies a scatterplot artist to show a beeswarm plot."""

    def __init__(self, orient='x', width=0.8, warn_thresh=0.05):
        self.orient = orient
        self.width = width
        self.warn_thresh = warn_thresh

    def __call__(self, points, center):
        """Swarm `points`, a PathCollection, around the `center` position."""
        ax = points.axes
        dpi = ax.figure.dpi
        orig_xy_data = points.get_offsets()
        cat_idx = 1 if self.orient == 'y' else 0
        orig_xy_data[:, cat_idx] = center
        orig_x_data, orig_y_data = orig_xy_data.T
        orig_xy = ax.transData.transform(orig_xy_data)
        if self.orient == 'y':
            orig_xy = orig_xy[:, [1, 0]]
        sizes = points.get_sizes()
        if sizes.size == 1:
            sizes = np.repeat(sizes, orig_xy.shape[0])
        edge = points.get_linewidth().item()
        radii = (np.sqrt(sizes) + edge) / 2 * (dpi / 72)
        orig_xy = np.c_[orig_xy, radii]
        sorter = np.argsort(orig_xy[:, 1])
        orig_xyr = orig_xy[sorter]
        new_xyr = np.empty_like(orig_xyr)
        new_xyr[sorter] = self.beeswarm(orig_xyr)
        if self.orient == 'y':
            new_xy = new_xyr[:, [1, 0]]
        else:
            new_xy = new_xyr[:, :2]
        new_x_data, new_y_data = ax.transData.inverted().transform(new_xy).T
        t_fwd, t_inv = _get_transform_functions(ax, self.orient)
        if self.orient == 'y':
            self.add_gutters(new_y_data, center, t_fwd, t_inv)
        else:
            self.add_gutters(new_x_data, center, t_fwd, t_inv)
        if self.orient == 'y':
            points.set_offsets(np.c_[orig_x_data, new_y_data])
        else:
            points.set_offsets(np.c_[new_x_data, orig_y_data])

    def beeswarm(self, orig_xyr):
        """Adjust x position of points to avoid overlaps."""
        midline = orig_xyr[0, 0]
        swarm = np.atleast_2d(orig_xyr[0])
        for xyr_i in orig_xyr[1:]:
            neighbors = self.could_overlap(xyr_i, swarm)
            candidates = self.position_candidates(xyr_i, neighbors)
            offsets = np.abs(candidates[:, 0] - midline)
            candidates = candidates[np.argsort(offsets)]
            new_xyr_i = self.first_non_overlapping_candidate(candidates, neighbors)
            swarm = np.vstack([swarm, new_xyr_i])
        return swarm

    def could_overlap(self, xyr_i, swarm):
        """Return a list of all swarm points that could overlap with target."""
        _, y_i, r_i = xyr_i
        neighbors = []
        for xyr_j in reversed(swarm):
            _, y_j, r_j = xyr_j
            if y_i - y_j < r_i + r_j:
                neighbors.append(xyr_j)
            else:
                break
        return np.array(neighbors)[::-1]

    def position_candidates(self, xyr_i, neighbors):
        """Return a list of coordinates that might be valid by adjusting x."""
        candidates = [xyr_i]
        x_i, y_i, r_i = xyr_i
        left_first = True
        for x_j, y_j, r_j in neighbors:
            dy = y_i - y_j
            dx = np.sqrt(max((r_i + r_j) ** 2 - dy ** 2, 0)) * 1.05
            cl, cr = ((x_j - dx, y_i, r_i), (x_j + dx, y_i, r_i))
            if left_first:
                new_candidates = [cl, cr]
            else:
                new_candidates = [cr, cl]
            candidates.extend(new_candidates)
            left_first = not left_first
        return np.array(candidates)

    def first_non_overlapping_candidate(self, candidates, neighbors):
        """Find the first candidate that does not overlap with the swarm."""
        if len(neighbors) == 0:
            return candidates[0]
        neighbors_x = neighbors[:, 0]
        neighbors_y = neighbors[:, 1]
        neighbors_r = neighbors[:, 2]
        for xyr_i in candidates:
            x_i, y_i, r_i = xyr_i
            dx = neighbors_x - x_i
            dy = neighbors_y - y_i
            sq_distances = np.square(dx) + np.square(dy)
            sep_needed = np.square(neighbors_r + r_i)
            good_candidate = np.all(sq_distances >= sep_needed)
            if good_candidate:
                return xyr_i
        raise RuntimeError('No non-overlapping candidates found. This should not happen.')

    def add_gutters(self, points, center, trans_fwd, trans_inv):
        """Stop points from extending beyond their territory."""
        half_width = self.width / 2
        low_gutter = trans_inv(trans_fwd(center) - half_width)
        off_low = points < low_gutter
        if off_low.any():
            points[off_low] = low_gutter
        high_gutter = trans_inv(trans_fwd(center) + half_width)
        off_high = points > high_gutter
        if off_high.any():
            points[off_high] = high_gutter
        gutter_prop = (off_high + off_low).sum() / len(points)
        if gutter_prop > self.warn_thresh:
            msg = '{:.1%} of the points cannot be placed; you may want to decrease the size of the markers or use stripplot.'.format(gutter_prop)
            warnings.warn(msg, UserWarning)
        return points