import numpy as np
import param
from ..core import Dataset, Dimension, NdOverlay
from ..core.operation import Operation
from ..core.util import cartesian_product, isfinite
from ..element import Area, Bivariate, Contours, Curve, Distribution, Image, Polygons
from .element import contours
def _kde_support(bin_range, bw, gridsize, cut, clip):
    """Establish support for a kernel density estimate."""
    kmin, kmax = (bin_range[0] - bw * cut, bin_range[1] + bw * cut)
    if isfinite(clip[0]):
        kmin = max(kmin, clip[0])
    if isfinite(clip[1]):
        kmax = min(kmax, clip[1])
    return np.linspace(kmin, kmax, gridsize)