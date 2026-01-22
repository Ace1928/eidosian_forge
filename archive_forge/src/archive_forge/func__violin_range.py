import numpy as np
from scipy.stats import gaussian_kde
from . import utils
def _violin_range(pos_data, plot_opts):
    """Return array with correct range, with which violins can be plotted."""
    cutoff = plot_opts.get('cutoff', False)
    cutoff_type = plot_opts.get('cutoff_type', 'std')
    cutoff_val = plot_opts.get('cutoff_val', 1.5)
    s = 0.0
    if not cutoff:
        if cutoff_type == 'std':
            s = cutoff_val * np.std(pos_data)
        else:
            s = cutoff_val
    x_lower = kde.dataset.min() - s
    x_upper = kde.dataset.max() + s
    return np.linspace(x_lower, x_upper, 100)