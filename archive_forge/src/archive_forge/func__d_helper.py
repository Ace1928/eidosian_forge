from collections import defaultdict
from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
from bokeh.models.annotations import Legend, Title
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram, kde
from ...plot_utils import _scale_fig_size, calculate_point_estimate, vectorized_to_hex
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
def _d_helper(vec, vname, color, bw, circular, line_width, markersize, hdi_prob, point_estimate, hdi_markers, outline, shade, ax):
    extra = {}
    plotted = []
    if vec.dtype.kind == 'f':
        if hdi_prob != 1:
            hdi_ = hdi(vec, hdi_prob, multimodal=False)
            new_vec = vec[(vec >= hdi_[0]) & (vec <= hdi_[1])]
        else:
            new_vec = vec
        x, density = kde(new_vec, circular=circular, bw=bw)
        density *= hdi_prob
        xmin, xmax = (x[0], x[-1])
        ymin, ymax = (density[0], density[-1])
        if outline:
            plotted.append(ax.line(x, density, line_color=color, line_width=line_width, **extra))
            plotted.append(ax.line([xmin, xmin], [-ymin / 100, ymin], line_color=color, line_dash='solid', line_width=line_width, muted_color=color, muted_alpha=0.2))
            plotted.append(ax.line([xmax, xmax], [-ymax / 100, ymax], line_color=color, line_dash='solid', line_width=line_width, muted_color=color, muted_alpha=0.2))
        if shade:
            plotted.append(ax.patch(np.r_[x[::-1], x, x[-1:]], np.r_[np.zeros_like(x), density, [0]], fill_color=color, fill_alpha=shade, muted_color=color, muted_alpha=0.2, **extra))
    else:
        xmin, xmax = hdi(vec, hdi_prob, multimodal=False)
        bins = get_bins(vec)
        _, hist, edges = histogram(vec, bins=bins)
        if outline:
            plotted.append(ax.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color=color, fill_color=None, muted_color=color, muted_alpha=0.2, **extra))
        else:
            plotted.append(ax.quad(top=hist, bottom=0, left=edges[:-1], right=edges[1:], line_color=color, fill_color=color, fill_alpha=shade, muted_color=color, muted_alpha=0.2, **extra))
    if hdi_markers:
        plotted.append(ax.diamond(xmin, 0, line_color='black', fill_color=color, size=markersize))
        plotted.append(ax.diamond(xmax, 0, line_color='black', fill_color=color, size=markersize))
    if point_estimate is not None:
        est = calculate_point_estimate(point_estimate, vec, bw, circular)
        plotted.append(ax.circle(est, 0, fill_color=color, line_color='black', size=markersize))
    _title = Title()
    _title.text = vname
    ax.title = _title
    ax.title.text_font_size = '13pt'
    return plotted