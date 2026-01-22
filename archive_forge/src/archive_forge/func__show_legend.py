import numpy as np
from scipy.stats import gaussian_kde
from . import utils
def _show_legend(ax):
    """Utility function to show legend."""
    leg = ax.legend(loc=1, shadow=True, fancybox=True, labelspacing=0.2, borderpad=0.15)
    ltext = leg.get_texts()
    llines = leg.get_lines()
    frame = leg.get_frame()
    from matplotlib.artist import setp
    setp(ltext, fontsize='small')
    setp(llines, linewidth=1)