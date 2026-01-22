import warnings
from matplotlib.image import _ImageBase
from matplotlib.patches import Patch
from matplotlib.transforms import Bbox, TransformedBbox, BboxTransform
import matplotlib as mpl
import numpy as np
from . import reductions
from . import transfer_functions as tf
from .colors import Sets1to3
from .core import bypixel, Canvas
def _binning(self, data, n=256):
    if np.ma.is_masked(data):
        data = data[~data.mask]
    low = data.min() if self.vmin is None else self.vmin
    high = data.max() if self.vmax is None else self.vmax
    nbins = self._nbins
    eq_bin_edges = np.linspace(low, high, nbins + 1)
    hist, _ = np.histogram(data, eq_bin_edges)
    eq_bin_centers = np.convolve(eq_bin_edges, [0.5, 0.5], mode='valid')
    cdf = np.cumsum(hist)
    cdf_max = cdf[-1]
    norm_cdf = cdf / cdf_max
    finite_bins = n - 1
    binning = []
    iterations = 0
    guess = n * 2
    while finite_bins != n and iterations < 4 and (finite_bins != 0):
        ratio = guess / finite_bins
        if ratio > 1000:
            break
        guess = np.round(max(n * ratio, n))
        palette_edges = np.arange(0, guess)
        palette_cdf = norm_cdf * (guess - 1)
        binning = np.interp(palette_edges, palette_cdf, eq_bin_centers)
        uniq_bins = np.unique(binning)
        finite_bins = len(uniq_bins) - 1
        iterations += 1
    if finite_bins == 0:
        binning = [low] + [high] * (n - 1)
    else:
        binning = binning[-n:]
        if finite_bins != n:
            warnings.warn('EqHistColorMapper warning: Histogram equalization did not converge.')
    return binning