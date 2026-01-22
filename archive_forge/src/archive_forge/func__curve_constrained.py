from statsmodels.compat.numpy import NP_LT_123
import numpy as np
from scipy.special import comb
from statsmodels.graphics.utils import _import_mpl
from statsmodels.multivariate.pca import PCA
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import itertools
from multiprocessing import Pool
from . import utils
def _curve_constrained(x, idx, sign, band, pca, ks_gaussian):
    """Find out if the curve is within the band.

    The curve value at :attr:`idx` for a given PDF is only returned if
    within bounds defined by the band. Otherwise, 1E6 is returned.

    Parameters
    ----------
    x : float
        Curve in reduced space.
    idx : int
        Index value of the components to compute.
    sign : int
        Return positive or negative value.
    band : list of float
        PDF values `[min_pdf, max_pdf]` to be within.
    pca : statsmodels Principal Component Analysis instance
        The PCA object to use.
    ks_gaussian : KDEMultivariate instance

    Returns
    -------
    value : float
        Curve value at `idx`.
    """
    x = x.reshape(1, -1)
    pdf = ks_gaussian.pdf(x)
    if band[0] < pdf < band[1]:
        value = sign * _inverse_transform(pca, x)[0][idx]
    else:
        value = 1000000.0
    return value