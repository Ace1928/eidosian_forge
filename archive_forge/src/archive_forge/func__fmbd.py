from statsmodels.compat.numpy import NP_LT_123
import numpy as np
from scipy.special import comb
from statsmodels.graphics.utils import _import_mpl
from statsmodels.multivariate.pca import PCA
from statsmodels.nonparametric.kernel_density import KDEMultivariate
import itertools
from multiprocessing import Pool
from . import utils
def _fmbd():
    down = rmat - 1
    up = n - rmat
    return (np.sum(up * down, axis=1) / p + n - 1) / comb(n, 2)