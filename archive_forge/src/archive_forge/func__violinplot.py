import numpy as np
from bokeh.models.annotations import Title
from ....stats import hdi
from ....stats.density_utils import get_bins, histogram, kde
from ...plot_utils import _scale_fig_size
from .. import show_layout
from . import backend_kwarg_defaults, create_axes_grid
def _violinplot(val, rug, side, shade, bw, circular, ax, **shade_kwargs):
    """Auxiliary function to plot violinplots."""
    if bw == 'default':
        bw = 'taylor' if circular else 'experimental'
    x, density = kde(val, circular=circular, bw=bw)
    if rug and side == 'both':
        side = 'right'
    if side == 'left':
        dens = -density
    elif side == 'right':
        x = x[::-1]
        dens = density[::-1]
    elif side == 'both':
        x = np.concatenate([x, x[::-1]])
        dens = np.concatenate([-density, density[::-1]])
    ax.harea(y=x, x1=dens, x2=np.zeros_like(dens), fill_alpha=shade, **shade_kwargs)
    return density