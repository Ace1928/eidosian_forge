import matplotlib.pyplot as plt
from matplotlib import _pylab_helpers
import numpy as np
from ....stats.density_utils import get_bins
from ...kdeplot import plot_kde
from ...plot_utils import _scale_fig_size, _init_kwargs_dict
from . import backend_kwarg_defaults, backend_show, create_axes_grid, matplotlib_kwarg_dealiaser
def _histplot_mpl_op(values, values2, rotated, ax, hist_kwargs, is_circular):
    """Add a histogram for the data to the axes."""
    bins = hist_kwargs.pop('bins', None)
    if is_circular == 'degrees':
        if bins is None:
            bins = get_bins(values)
        values = np.deg2rad(values)
        bins = np.deg2rad(bins)
    elif is_circular:
        labels = ['0', f'{np.pi / 4:.2f}', f'{np.pi / 2:.2f}', f'{3 * np.pi / 4:.2f}', f'{np.pi:.2f}', f'{-3 * np.pi / 4:.2f}', f'{-np.pi / 2:.2f}', f'{-np.pi / 4:.2f}']
        ax.set_xticklabels(labels)
    if values2 is not None:
        raise NotImplementedError('Insert hexbin plot here')
    if bins is None:
        bins = get_bins(values)
    if values.dtype.kind == 'i':
        hist_kwargs.setdefault('align', 'left')
    else:
        hist_kwargs.setdefault('align', 'mid')
    n, bins, _ = ax.hist(np.asarray(values).flatten(), bins=bins, **hist_kwargs)
    if values.dtype.kind == 'i':
        ticks = bins[:-1]
    else:
        ticks = (bins[1:] + bins[:-1]) / 2
    if rotated:
        ax.set_yticks(ticks)
    elif not is_circular:
        ax.set_xticks(ticks)
    if is_circular:
        ax.set_ylim(0, 1.5 * n.max())
        ax.set_yticklabels([])
    if hist_kwargs.get('label') is not None:
        ax.legend()
    return ax